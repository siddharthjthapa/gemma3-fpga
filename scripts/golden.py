#!/usr/bin/env python3
"""golden.py - host reference for the Gemma3-600K PL port.

Replicates gemma3_600k_rpi4/src/run.c forward() bit-faithfully in NumPy:
weights are fp16 cast to fp32; all math is fp32 (the C casts __fp16->float).
It (1) generates text with argmax to prove the model works, and (2) dumps
op-level golden vectors + logits at pos0/pos1 for the RTL test benches.

Run:  python golden.py        (from anywhere; paths are resolved relative to repo)
"""
import os, struct, sys
import numpy as np

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.normpath(os.path.join(HERE, ".."))
GOLD  = os.path.join(ROOT, "golden")
DUMP  = os.path.join(GOLD, "dump")
SIM   = os.path.join(ROOT, "sim")
os.makedirs(DUMP, exist_ok=True)
os.makedirs(SIM, exist_ok=True)

f32 = np.float32

# ---------------------------------------------------------------- config ----
CFG_FIELDS = ["dim","hidden_dim","n_layers","n_heads","n_kv_heads",
              "vocab_size","seq_len","head_dim","sliding_window",
              "full_attention_mask"]

def load_model(path):
    raw = open(path, "rb").read()
    cfg = dict(zip(CFG_FIELDS, struct.unpack("<10i", raw[:40])))
    # weights are fp16 starting right after the 40-byte config
    w16 = np.frombuffer(raw[40:], dtype=np.float16)
    return cfg, w16

def map_weights(cfg, w16):
    p = cfg
    q_dim  = p["n_heads"]    * p["head_dim"]
    kv_dim = p["n_kv_heads"] * p["head_dim"]
    L = p["n_layers"]
    w = {}
    off = [0]
    def take(n):
        s = off[0]; off[0] += n
        return w16[s:s+n].astype(f32)        # fp16 -> fp32 (exactly what the C does)
    w["token_embedding_table"] = take(p["vocab_size"] * p["dim"])
    w["rms_att_weight"]        = take(L * p["dim"])
    w["wq"]                    = take(L * q_dim * p["dim"])
    w["wk"]                    = take(L * kv_dim * p["dim"])
    w["wv"]                    = take(L * kv_dim * p["dim"])
    w["wo"]                    = take(L * p["dim"] * q_dim)
    w["q_norm"]                = take(L * p["head_dim"])
    w["k_norm"]                = take(L * p["head_dim"])
    w["post_att_weight"]       = take(L * p["dim"])
    w["rms_ffn_weight"]        = take(L * p["dim"])
    w["w1"]                    = take(L * p["hidden_dim"] * p["dim"])
    w["w2"]                    = take(L * p["dim"] * p["hidden_dim"])
    w["w3"]                    = take(L * p["hidden_dim"] * p["dim"])
    w["post_ffn_weight"]       = take(L * p["dim"])
    w["rms_final_weight"]      = take(p["dim"])
    w["wcls"]                  = w["token_embedding_table"]      # weight tying
    assert off[0] == len(w16), f"weight map mismatch: used {off[0]} of {len(w16)}"
    return w

# ----------------------------------------------------------- primitives -----
def rmsnorm(x, weight):
    # Gemma: out = x * (1/sqrt(mean(x^2)+1e-6)) * (1 + weight)
    x = x.astype(f32)
    ss = f32(np.dot(x, x))                       # sum of squares (fp32)
    ss = f32(1.0) / np.sqrt(ss / f32(x.size) + f32(1e-6))
    return (x * ss * (f32(1.0) + weight.astype(f32))).astype(f32)

def gelu_tanh(x):
    k0 = f32(0.7978845608); k1 = f32(0.044715)
    x = x.astype(f32)
    x3 = x * x * x
    return (f32(0.5) * x * (f32(1.0) + np.tanh(k0 * (x + k1 * x3)))).astype(f32)

def matmul(x, w, n, d):
    # out[i] = sum_j w[i*n+j]*x[j]   (sequential fp32, like the C)
    W = w.reshape(d, n).astype(f32)
    return (W @ x.astype(f32)).astype(f32)

def softmax(x):
    x = x.astype(f32)
    m = np.max(x)
    e = np.exp(x - m).astype(f32)
    return (e / np.sum(e)).astype(f32)

def build_rope(cfg):
    p = cfg
    half = p["head_dim"] // 2
    cos = np.zeros((2, p["seq_len"], half), dtype=f32)
    sin = np.zeros((2, p["seq_len"], half), dtype=f32)
    for tid in range(2):
        theta = f32(1000000.0) if tid else f32(10000.0)
        for pos in range(p["seq_len"]):
            for i in range(half):
                freq = f32(1.0) / np.power(theta, f32(2.0*i) / f32(p["head_dim"]))
                ang = f32(pos) * freq
                cos[tid, pos, i] = np.cos(ang)
                sin[tid, pos, i] = np.sin(ang)
    return cos, sin

def apply_rope(vec, n_heads, head_dim, cos_row, sin_row):
    half = head_dim // 2
    v = vec.copy()
    for h in range(n_heads):
        b = h * head_dim
        for i in range(half):
            a  = v[b + i]; bb = v[b + i + half]
            c  = cos_row[i]; s = sin_row[i]
            v[b + i]        = (a * c - bb * s)
            v[b + i + half] = (bb * c + a * s)
    return v.astype(f32)

# -------------------------------------------------------------- forward -----
class Runner:
    def __init__(self, cfg, w):
        self.p = cfg; self.w = w
        p = cfg
        self.kv_dim = p["n_kv_heads"] * p["head_dim"]
        self.q_dim  = p["n_heads"]    * p["head_dim"]
        self.kv_mul = p["n_heads"] // p["n_kv_heads"]
        self.cos, self.sin = build_rope(cfg)
        self.key_cache = np.zeros((p["n_layers"], p["seq_len"], self.kv_dim), dtype=f32)
        self.val_cache = np.zeros((p["n_layers"], p["seq_len"], self.kv_dim), dtype=f32)

    def is_full(self, layer):
        return (self.p["full_attention_mask"] & (1 << layer)) != 0

    def attn_start(self, layer, pos):
        if self.is_full(layer): return 0
        s = pos - self.p["sliding_window"] + 1
        return s if s > 0 else 0

    def forward(self, token, pos, dump=False):
        p = self.p; w = self.w
        dim = p["dim"]; hd = p["head_dim"]; half = hd // 2
        kv_dim = self.kv_dim; kv_mul = self.kv_mul
        embed_scale = np.sqrt(f32(dim))

        x = (w["token_embedding_table"][token*dim:(token+1)*dim] * embed_scale).astype(f32)
        if dump: dd("emb", x)

        for l in range(p["n_layers"]):
            xb = rmsnorm(x, w["rms_att_weight"][l*dim:(l+1)*dim])
            if dump and l == 0:
                dd("l0_attn_norm", xb)
                dd("rms_att_w", w["rms_att_weight"][l*dim:(l+1)*dim])
            q = matmul(xb, w["wq"][l*self.q_dim*dim:(l+1)*self.q_dim*dim], dim, self.q_dim)
            k = matmul(xb, w["wk"][l*kv_dim*dim:(l+1)*kv_dim*dim], dim, kv_dim)
            v = matmul(xb, w["wv"][l*kv_dim*dim:(l+1)*kv_dim*dim], dim, kv_dim)
            if dump and l == 0: dd("l0_q", q)

            # QK-norm (per head, weight length head_dim)
            qn = w["q_norm"][l*hd:(l+1)*hd]; kn = w["k_norm"][l*hd:(l+1)*hd]
            for h in range(p["n_heads"]):
                q[h*hd:(h+1)*hd] = rmsnorm(q[h*hd:(h+1)*hd], qn)
            for h in range(p["n_kv_heads"]):
                k[h*hd:(h+1)*hd] = rmsnorm(k[h*hd:(h+1)*hd], kn)
            if dump and l == 0: dd("l0_q_norm", q)

            tid = 1 if self.is_full(l) else 0
            q = apply_rope(q, p["n_heads"],    hd, self.cos[tid, pos], self.sin[tid, pos])
            k = apply_rope(k, p["n_kv_heads"], hd, self.cos[tid, pos], self.sin[tid, pos])
            if dump and l == 0: dd("l0_q_rope", q)

            self.key_cache[l, pos] = k
            self.val_cache[l, pos] = v

            xb_attn = np.zeros(self.q_dim, dtype=f32)
            inv = f32(1.0) / np.sqrt(f32(hd))
            start = self.attn_start(l, pos)
            for h in range(p["n_heads"]):
                qh = q[h*hd:(h+1)*hd]
                kvh = h // kv_mul
                scores = np.zeros(pos - start + 1, dtype=f32)
                for t in range(start, pos+1):
                    kt = self.key_cache[l, t, kvh*hd:(kvh+1)*hd]
                    scores[t-start] = f32(np.dot(qh, kt)) * inv
                a = softmax(scores)
                acc = np.zeros(hd, dtype=f32)
                for t in range(start, pos+1):
                    vt = self.val_cache[l, t, kvh*hd:(kvh+1)*hd]
                    acc += a[t-start] * vt
                xb_attn[h*hd:(h+1)*hd] = acc

            o = matmul(xb_attn, w["wo"][l*dim*self.q_dim:(l+1)*dim*self.q_dim], self.q_dim, dim)
            o = rmsnorm(o, w["post_att_weight"][l*dim:(l+1)*dim])   # post-attn norm
            x = (x + o).astype(f32)

            xb = rmsnorm(x, w["rms_ffn_weight"][l*dim:(l+1)*dim])
            hb  = matmul(xb, w["w1"][l*p["hidden_dim"]*dim:(l+1)*p["hidden_dim"]*dim], dim, p["hidden_dim"])
            hb2 = matmul(xb, w["w3"][l*p["hidden_dim"]*dim:(l+1)*p["hidden_dim"]*dim], dim, p["hidden_dim"])
            hb = (gelu_tanh(hb) * hb2).astype(f32)
            y = matmul(hb, w["w2"][l*dim*p["hidden_dim"]:(l+1)*dim*p["hidden_dim"]], p["hidden_dim"], dim)
            y = rmsnorm(y, w["post_ffn_weight"][l*dim:(l+1)*dim])   # post-ffn norm
            x = (x + y).astype(f32)
            if dump and l == 0: dd("x_after_l0", x)

        x = rmsnorm(x, w["rms_final_weight"])
        logits = matmul(x, w["wcls"], dim, p["vocab_size"])
        if dump: dd("logits", logits)
        return logits.astype(f32)

# --------------------------------------------------------------- dumps ------
def dd(name, data):
    data = np.asarray(data, dtype=f32)
    data.tofile(os.path.join(DUMP, name + ".bin"))
    print(f"  dump {name:14s} n={data.size:<6d} sum={float(np.sum(data)):.6f} [0]={float(data[0]):.6f}")

def whex(name, data):
    data = np.asarray(data, dtype=f32)
    with open(os.path.join(SIM, name), "w") as f:
        for v in data:
            f.write("%08x\n" % struct.unpack("<I", struct.pack("<f", v))[0])

# ----------------------------------------------------------- tokenizer ------
class Tok:
    def __init__(self, path):
        raw = open(path, "rb").read()
        (self.vocab_size,) = struct.unpack("<I", raw[:4])
        n = self.vocab_size
        self.offsets = np.frombuffer(raw[4:4+(n+1)*4], dtype="<u4").copy()
        self.bytes = raw[4+(n+1)*4:]
        self.max_len = int(np.max(np.diff(self.offsets)))
    def piece(self, i):
        return self.bytes[self.offsets[i]:self.offsets[i+1]]
    def encode(self, text, bos=True):
        b = text.encode("utf-8"); toks = []
        if bos: toks.append(2)
        pos = 0
        while pos < len(b):
            best_id, best_len = -1, 0
            for i in range(4, self.vocab_size):
                p = self.piece(i); ln = len(p)
                if ln > best_len and b[pos:pos+ln] == p:
                    best_id, best_len = i, ln
                    if best_len == self.max_len: break
            if best_id < 0: best_id, best_len = 3, 1
            toks.append(best_id); pos += best_len
        return toks
    def decode(self, i):
        return self.piece(i)

# --------------------------------------------------------------- main -------
def main():
    cfg, w16 = load_model(os.path.join(GOLD, "gemma3_600K.bin"))
    print("Config:", cfg)
    w = map_weights(cfg, w16)
    r = Runner(cfg, w)
    tok = Tok(os.path.join(GOLD, "gemma3_tok1024.bin"))
    print(f"Tokenizer: vocab={tok.vocab_size} max_piece={tok.max_len}")

    # ---- op-level dumps + logits at pos0 / pos1 for the RTL test bench -----
    # Use the real prompt's first two tokens so pos1 exercises attention over
    # two cached positions (BOS at pos0 would give a degenerate argmax=BOS).
    prompt = "The little robot lived in the red shed. Every morning it rolled across the floor. One day "
    toks = tok.encode(prompt, bos=True)
    t0, t1 = toks[0], toks[1]
    print(f"=== golden dumps (token={t0}, pos=0) ===")
    lg0 = r.forward(t0, 0, dump=True)
    a0 = int(np.argmax(lg0))
    whex("logits.hex", lg0)
    lg1 = r.forward(t1, 1)
    a1 = int(np.argmax(lg1))
    whex("logits_p1.hex", lg1)
    print(f"=== pos0(token={t0}) argmax={a0}  pos1(token={t1}) argmax={a1} ===")
    print(f"Prompt -> {len(toks)} tokens; generating (argmax)...")
    out = []
    token = toks[0]; steps = min(160, cfg["seq_len"])
    for pos in range(steps):
        lg = r.forward(token, pos)
        nxt = toks[pos+1] if pos < len(toks)-1 else int(np.argmax(lg))
        if nxt == 1: break
        if pos >= len(toks)-1:
            out.append(tok.decode(nxt))
        token = nxt
    text = b"".join(out).decode("utf-8", errors="replace")
    print("=== generation ===")
    print(prompt + text)
    print("=== done ===")

    # record expected argmax for the TB
    with open(os.path.join(SIM, "golden_meta.txt"), "w") as f:
        f.write(f"pos0_token={t0} pos0_argmax={a0}\n")
        f.write(f"pos1_token={t1} pos1_argmax={a1}\n")

if __name__ == "__main__":
    main()
