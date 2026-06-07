#!/usr/bin/env python3
"""Generate fp16 exp / rsqrt test vectors + references for tb_trans16."""
import os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); SIM=os.path.normpath(os.path.join(HERE,"..","sim"))
f16=np.float16
rng=np.random.default_rng(3)
# exp inputs: softmax args (<=0) and GELU sigmoid args (mixed), avoid overflow
ex=np.concatenate([rng.uniform(-12,0,40), rng.uniform(-4,4,24)]).astype(f16)
exr=np.exp(ex.astype(np.float32)).astype(f16)
# rsqrt inputs: rmsnorm mean+eps, softmax sum, gelu denom (positive)
rx=np.concatenate([rng.uniform(0.01,2,24), rng.uniform(2,250,24)]).astype(f16)
rxr=(1.0/np.sqrt(rx.astype(np.float32))).astype(f16)
def whex(fn,arr):
    with open(os.path.join(SIM,fn),"w") as f:
        for v in np.asarray(arr,dtype=f16).view(np.uint16): f.write("%04x\n"%v)
whex("exp_in.hex",ex); whex("exp_ref.hex",exr)
whex("rsq_in.hex",rx); whex("rsq_ref.hex",rxr)
with open(os.path.join(SIM,"trans_dims.txt"),"w") as f: f.write(f"{len(ex)} {len(rx)}\n")
print(f"exp vectors: {len(ex)}  rsqrt vectors: {len(rx)}")
