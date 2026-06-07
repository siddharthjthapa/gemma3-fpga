/* Tiny Gemma3-style Transformer inference for bare-metal AArch64/RPi4.
 *
 * Single-core runtime. Model weights and tokenizer are embedded in the ELF.
 * Architecture conventions are taken from Gemma3 base config, scaled down to
 * a 600K-class model: Gemma RMSNorm, QK norm, RoPE, GQA, sliding attention,
 * and GELU-tanh gated FFN.
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

extern unsigned char model_data_start[];
extern unsigned char model_data_end[];
extern unsigned char tokenizer_data_start[];
extern unsigned char tokenizer_data_end[];

char *strcpy(char *dst, const char *src) {
    char *d = dst;
    while ((*d++ = *src++) != '\0');
    return dst;
}

int strcmp(const char *s1, const char *s2) {
    while (*s1 && *s1 == *s2) { s1++; s2++; }
    return (unsigned char)*s1 - (unsigned char)*s2;
}

size_t strlen(const char *s) {
    const char *p = s;
    while (*p) p++;
    return p - s;
}

void *memset(void *dst, int c, size_t n) {
    unsigned char *d = (unsigned char *)dst;
    while (n--) *d++ = (unsigned char)c;
    return dst;
}

extern unsigned char _bare_heap_base asm ("heap_base");
static unsigned char *bump_ptr = 0;

static void *bare_malloc(size_t size) {
    if (!bump_ptr) bump_ptr = &_bare_heap_base;
    size = (size + 15) & ~(size_t)15;
    unsigned char *ptr = bump_ptr;
    bump_ptr += size;
    return ptr;
}

static void *bare_calloc(size_t count, size_t size) {
    size_t total = count * size;
    void *ptr = bare_malloc(total);
    memset(ptr, 0, total);
    return ptr;
}

static void bare_free(void *ptr) {
    (void)ptr;
}

#define malloc bare_malloc
#define calloc bare_calloc
#define free bare_free

#define SYS_TIMER_CLO  (*(volatile uint32_t *)0xFE003004)
#define SYS_TIMER_CHI  (*(volatile uint32_t *)0xFE003008)

static long time_in_ms(void) {
    uint32_t hi1, lo, hi2;
    do {
        hi1 = SYS_TIMER_CHI;
        lo = SYS_TIMER_CLO;
        hi2 = SYS_TIMER_CHI;
    } while (hi1 != hi2);
    return (long)((((uint64_t)hi1 << 32) | lo) / 1000);
}

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    int head_dim;
    int sliding_window;
    int full_attention_mask;
} Config;

typedef struct {
    __fp16 *token_embedding_table;
    __fp16 *rms_att_weight;
    __fp16 *wq;
    __fp16 *wk;
    __fp16 *wv;
    __fp16 *wo;
    __fp16 *q_norm;
    __fp16 *k_norm;
    __fp16 *post_att_weight;
    __fp16 *rms_ffn_weight;
    __fp16 *w1;
    __fp16 *w2;
    __fp16 *w3;
    __fp16 *post_ffn_weight;
    __fp16 *rms_final_weight;
    __fp16 *wcls;
} TransformerWeights;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    __fp16 *key_cache;
    __fp16 *value_cache;
    float *rope_cos;
    float *rope_sin;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
} Transformer;

static void malloc_run_state(RunState *s, Config *p) {
    int q_dim = p->n_heads * p->head_dim;
    int kv_dim = p->n_kv_heads * p->head_dim;
    int half = p->head_dim / 2;

    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(q_dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(__fp16));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(__fp16));
    s->rope_cos = calloc(p->seq_len * half * 2, sizeof(float));
    s->rope_sin = calloc(p->seq_len * half * 2, sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache || !s->rope_cos || !s->rope_sin) {
        printf("Gemma3 malloc failed\n");
        while (1);
    }

    for (int theta_id = 0; theta_id < 2; theta_id++) {
        float theta = theta_id ? 1000000.0f : 10000.0f;
        float *cos_base = s->rope_cos + theta_id * p->seq_len * half;
        float *sin_base = s->rope_sin + theta_id * p->seq_len * half;
        for (int pos = 0; pos < p->seq_len; pos++) {
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(theta, (2.0f * i) / (float)p->head_dim);
                float angle = pos * freq;
                cos_base[pos * half + i] = cosf(angle);
                sin_base[pos * half + i] = sinf(angle);
            }
        }
    }
}

static void memory_map_weights(TransformerWeights *w, Config *p, __fp16 *ptr) {
    int q_dim = p->n_heads * p->head_dim;
    int kv_dim = p->n_kv_heads * p->head_dim;
    unsigned long long n_layers = p->n_layers;

#define MAP(dst, n) do { (dst) = ptr; ptr += (n); } while (0)
    MAP(w->token_embedding_table, p->vocab_size * p->dim);
    MAP(w->rms_att_weight, n_layers * p->dim);
    MAP(w->wq, n_layers * q_dim * p->dim);
    MAP(w->wk, n_layers * kv_dim * p->dim);
    MAP(w->wv, n_layers * kv_dim * p->dim);
    MAP(w->wo, n_layers * p->dim * q_dim);
    MAP(w->q_norm, n_layers * p->head_dim);
    MAP(w->k_norm, n_layers * p->head_dim);
    MAP(w->post_att_weight, n_layers * p->dim);
    MAP(w->rms_ffn_weight, n_layers * p->dim);
    MAP(w->w1, n_layers * p->hidden_dim * p->dim);
    MAP(w->w2, n_layers * p->dim * p->hidden_dim);
    MAP(w->w3, n_layers * p->hidden_dim * p->dim);
    MAP(w->post_ffn_weight, n_layers * p->dim);
    MAP(w->rms_final_weight, p->dim);
    w->wcls = w->token_embedding_table;
#undef MAP
}

static void build_transformer(Transformer *t) {
    memcpy(&t->config, model_data_start, sizeof(Config));
    __fp16 *weights = (__fp16 *)(model_data_start + sizeof(Config));
    memory_map_weights(&t->weights, &t->config, weights);
    malloc_run_state(&t->state, &t->config);
}

static void rmsnorm(float *out, float *x, __fp16 *weight, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + 1e-6f);
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * (1.0f + (float)weight[i]);
}

static float gelu_tanh_scalar(float x) {
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x3)));
}

static void matmul(float *out, float *x, __fp16 *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        __fp16 *row = w + i * n;
        for (int j = 0; j < n; j++) val += (float)row[j] * x[j];
        out[i] = val;
    }
}

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < size; i++) x[i] *= inv;
}

static int is_full_attention_layer(Config *p, int layer) {
    return (p->full_attention_mask & (1 << layer)) != 0;
}

static int attention_start_pos(Config *p, int layer, int pos) {
    if (is_full_attention_layer(p, layer)) return 0;
    int start = pos - p->sliding_window + 1;
    return start > 0 ? start : 0;
}

static void apply_rope(float *vec, int n_heads, int head_dim, int pos,
                       float *cos_base, float *sin_base) {
    int half = head_dim / 2;
    float *rc = cos_base + pos * half;
    float *rs = sin_base + pos * half;
    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * head_dim;
        for (int i = 0; i < half; i++) {
            float a = v[i];
            float b = v[i + half];
            float c = rc[i];
            float s = rs[i];
            v[i] = a * c - b * s;
            v[i + half] = b * c + a * s;
        }
    }
}

static float *forward(Transformer *transformer, int token, int pos) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    int dim = p->dim;
    int head_dim = p->head_dim;
    int q_dim = p->n_heads * head_dim;
    int kv_dim = p->n_kv_heads * head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int half = head_dim / 2;
    float embed_scale = sqrtf((float)dim);

    __fp16 *row = w->token_embedding_table + token * dim;
    for (int i = 0; i < dim; i++) s->x[i] = (float)row[i] * embed_scale;

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, s->x, w->rms_att_weight + l * dim, dim);
        matmul(s->q, s->xb, w->wq + l * q_dim * dim, dim, q_dim);
        matmul(s->k, s->xb, w->wk + l * kv_dim * dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * kv_dim * dim, dim, kv_dim);

        for (int h = 0; h < p->n_heads; h++)
            rmsnorm(s->q + h * head_dim, s->q + h * head_dim,
                    w->q_norm + l * head_dim, head_dim);
        for (int h = 0; h < p->n_kv_heads; h++)
            rmsnorm(s->k + h * head_dim, s->k + h * head_dim,
                    w->k_norm + l * head_dim, head_dim);

        int theta_id = is_full_attention_layer(p, l) ? 1 : 0;
        float *cos_base = s->rope_cos + theta_id * p->seq_len * half;
        float *sin_base = s->rope_sin + theta_id * p->seq_len * half;
        apply_rope(s->q, p->n_heads, head_dim, pos, cos_base, sin_base);
        apply_rope(s->k, p->n_kv_heads, head_dim, pos, cos_base, sin_base);

        int loff = l * p->seq_len * kv_dim;
        for (int i = 0; i < kv_dim; i++) {
            s->key_cache[loff + pos * kv_dim + i] = (__fp16)s->k[i];
            s->value_cache[loff + pos * kv_dim + i] = (__fp16)s->v[i];
        }

        for (int h = 0; h < p->n_heads; h++) {
            float *qh = s->q + h * head_dim;
            float *att = s->att + h * p->seq_len;
            int kv_off = (h / kv_mul) * head_dim;
            int start = attention_start_pos(p, l, pos);
            int count = pos - start + 1;
            float inv_sqrt = 1.0f / sqrtf((float)head_dim);
            for (int t = start; t <= pos; t++) {
                __fp16 *kh = s->key_cache + loff + t * kv_dim + kv_off;
                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) score += qh[i] * (float)kh[i];
                att[t - start] = score * inv_sqrt;
            }
            softmax(att, count);
            for (int i = 0; i < head_dim; i++) qh[i] = 0.0f;
            for (int t = start; t <= pos; t++) {
                __fp16 *vh = s->value_cache + loff + t * kv_dim + kv_off;
                float a = att[t - start];
                for (int i = 0; i < head_dim; i++) qh[i] += a * (float)vh[i];
            }
        }

        matmul(s->xb2, s->q, w->wo + l * dim * q_dim, q_dim, dim);
        rmsnorm(s->xb2, s->xb2, w->post_att_weight + l * dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);
        matmul(s->hb, s->xb, w->w1 + l * p->hidden_dim * dim, dim, p->hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * p->hidden_dim * dim, dim, p->hidden_dim);
        for (int i = 0; i < p->hidden_dim; i++) {
            s->hb[i] = gelu_tanh_scalar(s->hb[i]) * s->hb2[i];
        }
        matmul(s->xb, s->hb, w->w2 + l * dim * p->hidden_dim, p->hidden_dim, dim);
        rmsnorm(s->xb, s->xb, w->post_ffn_weight + l * dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }

    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    matmul(s->logits, s->x, w->wcls, dim, p->vocab_size);
    return s->logits;
}

typedef struct {
    uint32_t vocab_size;
    uint32_t *offsets;
    unsigned char *bytes;
    int max_token_len;
} Tokenizer;

static void build_tokenizer(Tokenizer *tok) {
    unsigned char *ptr = tokenizer_data_start;
    memcpy(&tok->vocab_size, ptr, 4);
    ptr += 4;
    tok->offsets = (uint32_t *)ptr;
    ptr += (tok->vocab_size + 1) * sizeof(uint32_t);
    tok->bytes = ptr;
    tok->max_token_len = 1;
    for (uint32_t i = 0; i < tok->vocab_size; i++) {
        int len = (int)(tok->offsets[i + 1] - tok->offsets[i]);
        if (len > tok->max_token_len) tok->max_token_len = len;
    }
}

static int token_matches(Tokenizer *tok, int id, const unsigned char *text,
                         int pos, int text_len) {
    uint32_t start = tok->offsets[id];
    uint32_t end = tok->offsets[id + 1];
    int len = (int)(end - start);
    if (pos + len > text_len) return 0;
    unsigned char *piece = tok->bytes + start;
    for (int i = 0; i < len; i++) {
        if (piece[i] != text[pos + i]) return 0;
    }
    return len;
}

static void encode(Tokenizer *tok, char *text, int bos, int eos,
                   int *tokens, int *n_tokens) {
    unsigned char *bytes = (unsigned char *)text;
    int text_len = (int)strlen(text);
    int pos = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 2;
    while (pos < text_len) {
        int best_id = -1;
        int best_len = 0;
        for (uint32_t id = 4; id < tok->vocab_size; id++) {
            int len = token_matches(tok, (int)id, bytes, pos, text_len);
            if (len > best_len) {
                best_id = (int)id;
                best_len = len;
                if (best_len == tok->max_token_len) break;
            }
        }
        if (best_id < 0) {
            best_id = 3;
            best_len = 1;
        }
        tokens[(*n_tokens)++] = best_id;
        pos += best_len;
    }
    if (eos) tokens[(*n_tokens)++] = 1;
}

static char *decode(Tokenizer *tok, int token_id) {
    static char buf[256];
    if (token_id < 0 || (uint32_t)token_id >= tok->vocab_size) {
        buf[0] = '\0';
        return buf;
    }
    uint32_t start = tok->offsets[token_id];
    uint32_t end = tok->offsets[token_id + 1];
    uint32_t len = end - start;
    if (len >= sizeof(buf)) len = sizeof(buf) - 1;
    memcpy(buf, tok->bytes + start, len);
    buf[len] = '\0';
    return buf;
}

static void safe_printf(char *piece) {
    if (!piece || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char b = (unsigned char)piece[0];
        if (!(isprint(b) || isspace(b))) return;
    }
    printf("%s", piece);
}

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

static unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

static float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample_argmax(float *probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

static int compare_prob(const void *a, const void *b) {
    const ProbIndex *pa = (const ProbIndex *)a;
    const ProbIndex *pb = (const ProbIndex *)b;
    if (pa->prob > pb->prob) return -1;
    if (pa->prob < pb->prob) return 1;
    return 0;
}

static int sample_topp(float *probabilities, int n, float topp,
                       ProbIndex *probindex, float coin) {
    int n0 = 0;
    float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_prob);
    float cumulative = 0.0f;
    int last = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative += probindex[i].prob;
        if (cumulative > topp) {
            last = i;
            break;
        }
    }
    float r = coin * cumulative;
    float cdf = 0.0f;
    for (int i = 0; i <= last; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) return probindex[i].index;
    }
    return probindex[last].index;
}

static int sample(Sampler *sampler, float *logits) {
    if (sampler->temperature == 0.0f) return sample_argmax(logits, sampler->vocab_size);
    for (int i = 0; i < sampler->vocab_size; i++) logits[i] /= sampler->temperature;
    softmax(logits, sampler->vocab_size);
    return sample_topp(logits, sampler->vocab_size, sampler->topp,
                       sampler->probindex, random_f32(&sampler->rng_state));
}

static void generate(Transformer *transformer, Tokenizer *tokenizer,
                     Sampler *sampler, char *prompt, int steps) {
    int *prompt_tokens = malloc((strlen(prompt) + 4) * sizeof(int));
    int num_prompt_tokens = 0;
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        printf("prompt encoding failed\n");
        while (1);
    }

    int token = prompt_tokens[0];
    int pos = 0;
    int next = 0;
    long start = 0;
    while (pos < steps) {
        float *logits = forward(transformer, token, pos);
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;
        if (next == 1) break;
        if (pos >= num_prompt_tokens) safe_printf(decode(tokenizer, next));
        token = next;
        if (start == 0 && pos >= num_prompt_tokens) start = time_in_ms();
    }
    printf("\n");
    if (start != 0 && pos > num_prompt_tokens + 1) {
        long elapsed = time_in_ms() - start;
        if (elapsed > 0) {
            int generated = pos - num_prompt_tokens;
            int toks = (int)((long)generated * 1000 / elapsed);
            printf("\nGeneration speed: %d tokens/second\n", toks);
        }
    }
    free(prompt_tokens);
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    printf("Tiny Gemma3 bare metal inference on RPi 4\n");
    printf("Building transformer from embedded model...\n");

    Transformer transformer;
    build_transformer(&transformer);

    Config *cfg = &transformer.config;
    printf("Model: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d "
           "head_dim=%d vocab=%d seq=%d sliding=%d\n",
           cfg->dim, cfg->hidden_dim, cfg->n_layers, cfg->n_heads,
           cfg->n_kv_heads, cfg->head_dim, cfg->vocab_size,
           cfg->seq_len, cfg->sliding_window);

    printf("Building tokenizer...\n");
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer);
    printf("Tokenizer: vocab=%d max_piece=%d\n",
           tokenizer.vocab_size, tokenizer.max_token_len);

    Sampler sampler;
    sampler.vocab_size = cfg->vocab_size;
    sampler.temperature = 0.35f;
    sampler.topp = 0.85f;
    sampler.rng_state = 42;
    sampler.probindex = malloc(sampler.vocab_size * sizeof(ProbIndex));

    char *prompt = "The little robot lived in the red shed. Every morning it rolled across the floor. One day ";
    int steps = 256;
    if (steps > cfg->seq_len) steps = cfg->seq_len;

    printf("Generating with prompt: \"%s\"\n---\n%s", prompt, prompt);
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
    printf("---\nDone.\n");
    return 0;
}
