#!/usr/bin/env python3
"""gen_blob.py - build the fp16 DDR weight blob for the Gemma3-600K PL core.

Reads golden/gemma3_600K.bin (fp16 weights), and emits, in the exact offset
order the RTL streams them:
  * matmul matrices (wq,wk,wv,wo,w1,w2,w3 and the tied classifier) ROW-INTERLEAVED
    by 4  -> one 64-bit AXI beat (= 4 fp16) feeds 4 output lanes sharing x[j];
  * norm-weight / embedding / RoPE vectors plain (row-major);
  * RoPE cos/sin tables (theta=10000, fp16) precomputed and baked in.

Outputs (all fp16):
  sim/model.hex   one 16-bit hex word per line  ($readmemh into a [15:0] mem)
  soc/model.bin   raw little-endian fp16        (downloaded to DDR over JTAG)
Prints the O_* halfword offsets to paste into the RTL localparams.
"""
import os, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
GOLD = os.path.join(ROOT, "golden")
SIM  = os.path.join(ROOT, "sim")
SOC  = os.path.join(ROOT, "soc")
for d in (SIM, SOC): os.makedirs(d, exist_ok=True)

CFG = ["dim","hidden_dim","n_layers","n_heads","n_kv_heads","vocab_size",
       "seq_len","head_dim","sliding_window","full_attention_mask"]

def load():
    raw = open(os.path.join(GOLD, "gemma3_600K.bin"), "rb").read()
    cfg = dict(zip(CFG, struct.unpack("<10i", raw[:40])))
    w16 = np.frombuffer(raw[40:], dtype=np.float16).copy()
    return cfg, w16

def interleave4(block, N, D):
    """block: D*N row-major (W[d][n] at d*N+n) -> groups of 4 rows:
       W[4p][n],W[4p+1][n],W[4p+2][n],W[4p+3][n] for each n. Pure permutation."""
    assert D % 4 == 0 and len(block) == N*D, (len(block), N, D)
    out = np.empty_like(block)
    o = 0
    for p in range(D//4):
        r = [(4*p+e)*N for e in range(4)]
        for n in range(N):
            for e in range(4):
                out[o] = block[r[e]+n]; o += 1
    assert np.array_equal(np.sort(out.view(np.uint16)), np.sort(block.view(np.uint16)))
    return out

def main():
    cfg, w16 = load()
    p = cfg
    dim, hid, NL = p["dim"], p["hidden_dim"], p["n_layers"]
    NH, NKV, HD = p["n_heads"], p["n_kv_heads"], p["head_dim"]
    VOC, SEQ = p["vocab_size"], p["seq_len"]
    q_dim, kv_dim, half = NH*HD, NKV*HD, HD//2
    print("Config:", cfg)

    # ---- slice the source blob in its native order -------------------------
    off = [0]
    def take(n):
        s = off[0]; off[0] += n; return w16[s:s+n].copy()
    emb       = take(VOC*dim)
    rms_att   = take(NL*dim)
    wq        = take(NL*q_dim*dim)
    wk        = take(NL*kv_dim*dim)
    wv        = take(NL*kv_dim*dim)
    wo        = take(NL*dim*q_dim)
    q_norm    = take(NL*HD)
    k_norm    = take(NL*HD)
    post_att  = take(NL*dim)
    rms_ffn   = take(NL*dim)
    w1        = take(NL*hid*dim)
    w2        = take(NL*dim*hid)
    w3        = take(NL*hid*dim)
    post_ffn  = take(NL*dim)
    rms_final = take(dim)
    assert off[0] == len(w16), (off[0], len(w16))

    # ---- interleave-by-4 each matmul matrix (per layer) --------------------
    def il_layers(buf, N, D):
        out = []
        for l in range(NL):
            out.append(interleave4(buf[l*N*D:(l+1)*N*D], N, D))
        return np.concatenate(out)
    wq_i = il_layers(wq, dim, q_dim)
    wk_i = il_layers(wk, dim, kv_dim)
    wv_i = il_layers(wv, dim, kv_dim)
    wo_i = il_layers(wo, q_dim, dim)
    w1_i = il_layers(w1, dim, hid)
    w2_i = il_layers(w2, hid, dim)
    w3_i = il_layers(w3, dim, hid)
    wcls_i = interleave4(emb.copy(), dim, VOC)        # classifier = tied emb, interleaved

    # ---- RoPE tables (theta=10000 local; mask=0 so global table unused) -----
    # Stride padded to RST=12 (multiple of 4) so every per-position read
    # (O_FREAL + pos*RST) lands on an 8-byte AXI boundary. With stride=half=10,
    # odd positions are only 4-byte aligned and the real HP aligns the address
    # DOWN to 8 bytes -> shifted RoPE -> wrong K/Q on odd positions (HW-only bug).
    RST = 12
    cos = np.zeros(SEQ*RST, dtype=np.float16)
    sin = np.zeros(SEQ*RST, dtype=np.float16)
    theta = np.float32(10000.0)
    for pos in range(SEQ):
        for i in range(half):
            freq = np.float32(1.0)/np.power(theta, np.float32(2.0*i)/np.float32(HD))
            ang = np.float32(pos)*freq
            cos[pos*RST+i] = np.float16(np.cos(ang))
            sin[pos*RST+i] = np.float16(np.sin(ang))

    # ---- assemble the blob, recording halfword offsets ---------------------
    blob = []
    offs = {}
    def add(name, arr):
        offs[name] = sum(len(a) for a in blob)
        blob.append(np.asarray(arr, dtype=np.float16))
    add("O_TOK",     emb)          # row-major (embedding lookup)
    add("O_RMSA",    rms_att)
    add("O_WQ",      wq_i)
    add("O_WK",      wk_i)
    add("O_WV",      wv_i)
    add("O_WO",      wo_i)
    add("O_QNORM",   q_norm)
    add("O_KNORM",   k_norm)
    add("O_POSTATT", post_att)
    add("O_RMSF",    rms_ffn)
    add("O_W1",      w1_i)
    add("O_W2",      w2_i)
    add("O_W3",      w3_i)
    add("O_POSTFFN", post_ffn)
    add("O_RMSFIN",  rms_final)
    add("O_FREAL",   cos)
    add("O_FIMAG",   sin)
    add("O_WCLS",    wcls_i)
    blob = np.concatenate(blob)
    total = len(blob)

    # ---- write sim/model.hex (16-bit words) and soc/model.bin (raw fp16) ----
    u = blob.view(np.uint16)
    with open(os.path.join(SIM, "model.hex"), "w") as f:
        f.write("\n".join("%04x" % v for v in u) + "\n")
    blob.tofile(os.path.join(SOC, "model.bin"))

    print(f"blob: {total} fp16 words = {total*2} bytes")
    print("RTL localparams (halfword offsets):")
    print("    localparam int " + ", ".join(f"{k}={v}" for k, v in offs.items()) + ";")
    print(f"    localparam int O_END={total};")
    # also dump offsets to a file the soc/sim can read
    with open(os.path.join(SIM, "blob_offsets.txt"), "w") as f:
        for k, v in offs.items(): f.write(f"{k} {v}\n")
        f.write(f"O_END {total}\n")

if __name__ == "__main__":
    main()
