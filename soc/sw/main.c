/* main.c - bare-metal driver for the Gemma3-600K PL accelerator (Z-turn Lite).
 *
 * The PL runs the whole fp16 Gemma3 transformer forward pass at 100 MHz,
 * streaming fp16 weights from DDR over its AXI HP master. The PS (Cortex-A9)
 * only: pokes token/pos + start over AXI-Lite, polls done, reads the 1024 fp16
 * logits back, does greedy argmax, decodes via vocab.h, and prints over UART1.
 *
 * The fp16 weight blob (model.bin, produced by script/gen_blob.py) must be in
 * DDR at WEIGHTS_DDR before running (the run script downloads it over JTAG).
 */
#include "xil_io.h"
#include "xil_printf.h"
#include "vocab.h"

#define GT_LO   0xF8F00200u
#define GT_HI   0xF8F00204u
#define GT_CTRL 0xF8F00208u
#define GT_HZ   333333333u
static u64 gt_now(void) {
    u32 hi, lo, hi2;
    do { hi = Xil_In32(GT_HI); lo = Xil_In32(GT_LO); hi2 = Xil_In32(GT_HI); } while (hi != hi2);
    return ((u64)hi << 32) | lo;
}

#define ACC_BASE     0x40000000u
#define R_CTRL       (ACC_BASE + 0x000u)
#define R_STATUS     (ACC_BASE + 0x004u)
#define R_TOKEN      (ACC_BASE + 0x008u)
#define R_POS        (ACC_BASE + 0x00Cu)
#define R_WBASE      (ACC_BASE + 0x010u)
#define R_LOGITS     (ACC_BASE + 0x1000u)   /* fp16 logit[k] (low 16b) at +4*k */

#define WEIGHTS_DDR  0x10000000u
#define MAXPOS       128                     /* generate up to this many tokens  */
#define BOS          2
#define EOS          1

/* fp16 (1/5/10, bias 15) bit pattern -> float */
static float h2f(u16 h) {
    u32 s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, out;
    if (e == 0) {
        if (m == 0) out = s << 31;
        else { int ee = -1; do { m <<= 1; ee++; } while (!(m & 0x400u));
               m &= 0x3FFu; out = (s << 31) | ((u32)(127 - 15 - ee) << 23) | (m << 13); }
    } else if (e == 0x1F) {
        out = (s << 31) | (0xFFu << 23) | (m << 13);
    } else {
        out = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    }
    union { u32 u; float f; } c; c.u = out; return c.f;
}

static int run_token(u16 tok, u16 pos) {
    Xil_Out32(R_TOKEN, tok);
    Xil_Out32(R_POS,   pos);
    Xil_Out32(R_CTRL,  1u);
    while ((Xil_In32(R_STATUS) & 1u) == 0u) { }
    int mi = 0; float mx = -1e30f;
    for (int k = 0; k < VOCAB_SIZE; k++) {
        float v = h2f((u16)(Xil_In32(R_LOGITS + 4u * k) & 0xFFFFu));
        if (k == 0 || v > mx) { mx = v; mi = k; }
    }
    return mi;
}

static void emit(int tok) {
    const char *p = vocab[tok];
    int n = vocab_len[tok];
    for (int j = 0; j < n; j++) {
        unsigned char c = (unsigned char)p[j];
        if (c == '\n' || c == '\t' || (c >= 0x20 && c < 0x7f)) xil_printf("%c", c);
    }
}

int main(void) {
    Xil_Out32(R_WBASE, WEIGHTS_DDR);
    xil_printf("\r\n=== Gemma3-600K transformer on Z-turn PL fabric @100MHz ===\r\n");
    xil_printf("(hand-written fp16 SystemVerilog forward pass; PS only samples)\r\n\r\n");
    xil_printf("%s", prompt_text);

    Xil_Out32(GT_CTRL, 1u);
    u64 t0 = gt_now(), tfirst = t0, t1;

    int tok = prompt_tokens[0], pos = 0, next;
    while (pos < MAXPOS) {
        next = run_token((u16)tok, (u16)pos);
        if (pos == 0) tfirst = gt_now();
        if (pos < N_PROMPT - 1) next = prompt_tokens[pos + 1];  /* teacher-force the prompt */
        pos++;
        if (next == EOS) break;
        if (pos >= N_PROMPT) emit(next);
        tok = next;
    }
    t1 = gt_now();

    u32 us_total = (u32)((t1 - t0)     * 1000000u / GT_HZ);
    u32 us_first = (u32)((tfirst - t0) * 1000000u / GT_HZ);
    u32 tok_milli = (u32)((u64)pos * 1000u * GT_HZ / (t1 - t0));
    xil_printf("\r\n\r\n=== %d tokens in %u us  (1st tok %u us) ===\r\n", pos, us_total, us_first);
    xil_printf("=== throughput = %u.%03u tokens/sec ===\r\n", tok_milli / 1000u, tok_milli % 1000u);
    for (;;) { }
    return 0;
}
