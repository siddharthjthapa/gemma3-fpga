#!/usr/bin/env python3
"""prec_probe.py - does Gemma3-600K survive fp16 compute?  (vectorized)

Compares three precision modes against the fp32 reference, modeling fp16
dot-products the way the real `pmac` hardware reduces them: 8 round-robin
accumulator buckets + a 3-level tree reduce, every step rounded to fp16.
  fp32  : reference (fp32 products + fp32 accumulate)
  mixed : fp16 products, fp32 accumulate (fp16 multipliers, fp32 accumulators)
  fp16  : fp16 products AND fp16 accumulate (a pure-fp16 datapath)
Weights and KV cache are fp16 in every mode.
"""
import os
import numpy as np
from golden import load_model, map_weights, build_rope, Tok, GOLD

f32 = np.float32
def h(x): return np.float32(np.float16(x))   # round to fp16 grid, fp32 container

PREC = "fp32"

def pmac_dot(W, x):
    """Row-wise dot of W:(d,n) with x:(n,), modeling the hardware reduction.
       fp32: plain fp32. mixed: fp16 products, fp32 sum. fp16: 8-bucket fp16."""
    if PREC == "fp32":
        return (W.astype(f32) @ x.astype(f32)).astype(f32)
    P = h(h(W) * h(x))                         # fp16 products
    if PREC == "mixed":
        return P.astype(f32).sum(axis=1).astype(f32)
    # full fp16: 8 round-robin buckets, sequential fp16 accumulate per bucket
    d, n = P.shape
    buck = np.zeros((8, d), dtype=f32)
    for j in range(n):
        buck[j & 7] = h(buck[j & 7] + P[:, j])
    r = buck
    r = np.stack([h(r[0]+r[1]), h(r[2]+r[3]), h(r[4]+r[5]), h(r[6]+r[7])])
    r = np.stack([h(r[0]+r[1]), h(r[2]+r[3])])
    return h(r[0] + r[1])

def matmul(x, w, n, d):
    return pmac_dot(w.reshape(d, n), x)

def vdot(a, b):
    return float(pmac_dot(np.asarray(a, f32)[None, :], np.asarray(b, f32))[0])

def rmsnorm(x, weight):
    x = x.astype(f32)
    ss = vdot(x, x)
    inv = f32(1.0) / np.sqrt(f32(ss) / f32(x.size) + f32(1e-6))
    if PREC != "fp32": inv = h(inv)
    g = (f32(1.0) + (h(weight) if PREC != "fp32" else weight.astype(f32)))
    if PREC == "fp32":
        return (x * inv * g).astype(f32)
    return h(h(x * inv) * g)

def gelu_tanh(x):
    k0 = f32(0.7978845608); k1 = f32(0.044715); x = x.astype(f32)
    if PREC == "fp32":
        x3 = x*x*x
        return (f32(0.5)*x*(f32(1.0)+np.tanh(k0*(x+k1*x3)))).astype(f32)
    x3 = h(h(h(x*x)*x))
    inner = h(k0 * h(x + h(k1*x3)))
    t = h(np.tanh(inner))
    return h(h(f32(0.5)*x) * (f32(1.0)+t))

def softmax(x):
    x = x.astype(f32); m = np.max(x)
    if PREC == "fp32":
        e = np.exp(x-m).astype(f32); return (e/np.sum(e)).astype(f32)
    e = h(np.exp(h(x-m)))
    s = f32(0.0)
    for v in e: s = h(s+v)
    return h(e * h(f32(1.0)/s))

class R:
    def __init__(self, cfg, w):
        self.p=cfg; self.w=w
        self.kv_dim=cfg["n_kv_heads"]*cfg["head_dim"]; self.q_dim=cfg["n_heads"]*cfg["head_dim"]
        self.kv_mul=cfg["n_heads"]//cfg["n_kv_heads"]; self.cos,self.sin=build_rope(cfg)
        self.kc=np.zeros((cfg["n_layers"],cfg["seq_len"],self.kv_dim),dtype=f32)
        self.vc=np.zeros((cfg["n_layers"],cfg["seq_len"],self.kv_dim),dtype=f32)
    def start(self,pos):
        s=pos-self.p["sliding_window"]+1; return s if s>0 else 0
    def rope(self,vec,nh,hd,cr,sr):
        half=hd//2; v=vec.copy().astype(f32)
        for hh in range(nh):
            b=hh*hd
            for i in range(half):
                a=v[b+i]; bb=v[b+i+half]; c=cr[i]; s=sr[i]
                if PREC=="fp32":
                    v[b+i]=a*c-bb*s; v[b+i+half]=bb*c+a*s
                else:
                    v[b+i]=h(h(a*c)-h(bb*s)); v[b+i+half]=h(h(bb*c)+h(a*s))
        return v
    def fwd(self, token, pos):
        p=self.p; w=self.w; dim=p["dim"]; hd=p["head_dim"]
        es=np.sqrt(f32(dim)); es=h(es) if PREC!="fp32" else es
        row=w["token_embedding_table"][token*dim:(token+1)*dim]
        x=(h(row*es) if PREC!="fp32" else (row*es)).astype(f32)
        if getattr(self,"dbg",False): self.d_emb=x.copy()
        for l in range(p["n_layers"]):
            xb=rmsnorm(x, w["rms_att_weight"][l*dim:(l+1)*dim])
            if getattr(self,"dbg",False) and l==0: self.d_xb=xb.copy()
            q=matmul(xb, w["wq"][l*self.q_dim*dim:(l+1)*self.q_dim*dim], dim, self.q_dim)
            k=matmul(xb, w["wk"][l*self.kv_dim*dim:(l+1)*self.kv_dim*dim], dim, self.kv_dim)
            v=matmul(xb, w["wv"][l*self.kv_dim*dim:(l+1)*self.kv_dim*dim], dim, self.kv_dim)
            if getattr(self,"dbg",False) and l==0: self.d_q=q.copy(); self.d_v=v.copy()
            qn=w["q_norm"][l*hd:(l+1)*hd]; kn=w["k_norm"][l*hd:(l+1)*hd]
            for hh in range(p["n_heads"]): q[hh*hd:(hh+1)*hd]=rmsnorm(q[hh*hd:(hh+1)*hd],qn)
            for hh in range(p["n_kv_heads"]): k[hh*hd:(hh+1)*hd]=rmsnorm(k[hh*hd:(hh+1)*hd],kn)
            q=self.rope(q,p["n_heads"],hd,self.cos[0,pos],self.sin[0,pos])
            k=self.rope(k,p["n_kv_heads"],hd,self.cos[0,pos],self.sin[0,pos])
            self.kc[l,pos]=h(k) if PREC!="fp32" else k     # KV cache is fp16
            self.vc[l,pos]=h(v) if PREC!="fp32" else v
            xba=np.zeros(self.q_dim,dtype=f32); inv=f32(1.0)/np.sqrt(f32(hd))
            inv=h(inv) if PREC!="fp32" else inv; st=self.start(pos)
            for hh in range(p["n_heads"]):
                qh=q[hh*hd:(hh+1)*hd]; kvh=hh//self.kv_mul
                K=self.kc[l,st:pos+1,kvh*hd:(kvh+1)*hd]              # (T,hd)
                sc=pmac_dot(K, qh)
                sc=(h(sc*inv) if PREC!="fp32" else sc*inv).astype(f32)
                a=softmax(sc)
                V=self.vc[l,st:pos+1,kvh*hd:(kvh+1)*hd]              # (T,hd)
                xba[hh*hd:(hh+1)*hd]=pmac_dot(V.T, a)               # sum_t a_t V_t
            if getattr(self,"dbg",False) and l==0: self.d_attn=xba.copy()
            o=matmul(xba, w["wo"][l*dim*self.q_dim:(l+1)*dim*self.q_dim], self.q_dim, dim)
            if getattr(self,"dbg",False) and l==0: self.d_wo=o.copy()
            o=rmsnorm(o, w["post_att_weight"][l*dim:(l+1)*dim])
            if getattr(self,"dbg",False) and l==0: self.d_o=o.copy()
            x=(h(x+o) if PREC!="fp32" else x+o).astype(f32)
            if getattr(self,"dbg",False) and l==0: self.d_xatt=x.copy()
            xb=rmsnorm(x, w["rms_ffn_weight"][l*dim:(l+1)*dim])
            hb=matmul(xb, w["w1"][l*p["hidden_dim"]*dim:(l+1)*p["hidden_dim"]*dim], dim, p["hidden_dim"])
            hb2=matmul(xb, w["w3"][l*p["hidden_dim"]*dim:(l+1)*p["hidden_dim"]*dim], dim, p["hidden_dim"])
            g=gelu_tanh(hb); hb=(h(g*hb2) if PREC!="fp32" else g*hb2).astype(f32)
            if getattr(self,"dbg",False) and l==0: self.d_hb=hb.copy()
            y=matmul(hb, w["w2"][l*dim*p["hidden_dim"]:(l+1)*dim*p["hidden_dim"]], p["hidden_dim"], dim)
            y=rmsnorm(y, w["post_ffn_weight"][l*dim:(l+1)*dim])
            x=(h(x+y) if PREC!="fp32" else x+y).astype(f32)
            if getattr(self,"dbg",False) and l==0: self.d_xL0=x.copy()
        x=rmsnorm(x, w["rms_final_weight"])
        return matmul(x, w["wcls"], dim, p["vocab_size"])

def gen(cfg, w, tok, mode, steps=80):
    global PREC; PREC=mode; r=R(cfg,w)
    prompt="The little robot lived in the red shed. Every morning it rolled across the floor. One day "
    toks=tok.encode(prompt,bos=True); out=[]; args=[]; token=toks[0]
    for pos in range(steps):
        lg=r.fwd(token,pos); am=int(np.argmax(lg)); args.append(am)
        nxt=toks[pos+1] if pos<len(toks)-1 else am
        if pos>=len(toks)-1:
            if nxt==1: break
            out.append(tok.decode(nxt))
        token=nxt
    return prompt+b"".join(out).decode("utf-8",errors="replace"), args

SIM=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","sim"))
def whex16(fn,arr):
    with open(os.path.join(SIM,fn),"w") as f:
        for v in np.asarray(arr,dtype=np.float16).view(np.uint16): f.write("%04x\n"%v)

def emit16(cfg,w,tok):
    """Emit the fp16-model golden logits the RTL must reproduce (argmax-faithful)."""
    global PREC; PREC="fp16"; r=R(cfg,w)
    t0=386
    lg0=r.fwd(t0,0); a0=int(np.argmax(lg0)); whex16("logits_fp16.hex",lg0)
    t1=a0
    lg1=r.fwd(t1,1); a1=int(np.argmax(lg1)); whex16("logits_p1_fp16.hex",lg1)
    with open(os.path.join(SIM,"golden16_meta.txt"),"w") as f:
        f.write(f"pos0_token={t0} pos0_argmax={a0}\n")
        f.write(f"pos1_token={t1} pos1_argmax={a1}\n")
    print(f"emit16: pos0 token={t0} argmax={a0}; pos1 token={t1} argmax={a1}")
    return t0,a0,t1,a1

def main():
    cfg,w16=load_model(os.path.join(GOLD,"gemma3_600K.bin"))
    w=map_weights(cfg,w16); tok=Tok(os.path.join(GOLD,"gemma3_tok1024.bin"))
    import sys
    if "--emit" in sys.argv:
        emit16(cfg,w,tok)
        # op-level fp16 dumps for token=386 pos=0 (RTL bring-up)
        global PREC; PREC="fp16"; rr=R(cfg,w); rr.dbg=True; rr.fwd(386,0)
        whex16("op_emb.hex", rr.d_emb); whex16("op_xb.hex", rr.d_xb)
        whex16("op_q.hex", rr.d_q);     whex16("op_xL0.hex", rr.d_xL0)
        whex16("op_xatt.hex", rr.d_xatt); whex16("op_hb.hex", rr.d_hb)
        whex16("op_v.hex", rr.d_v); whex16("op_attn.hex", rr.d_attn)
        whex16("op_wo.hex", rr.d_wo); whex16("op_o.hex", rr.d_o)
        print("op dumps written")
        return
    steps=80
    txt32,a32=gen(cfg,w,tok,"fp32",steps)
    print("\n========== fp32 (reference) ==========\n"+txt32)
    for mode in ("mixed","fp16"):
        txt,a=gen(cfg,w,tok,mode,steps)
        n=min(len(a),len(a32)); agree=sum(1 for i in range(n) if a[i]==a32[i])
        print(f"\n========== {mode} ==========  argmax agreement vs fp32: {agree}/{n}")
        print(txt)
    emit16(cfg,w,tok)

if __name__=="__main__":
    main()
