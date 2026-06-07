#!/usr/bin/env python3
"""Generate fp16 dot-product test cases for tb_pmac16.
   Reference uses the EXACT pmac16 reduction: 8 round-robin fp16 buckets
   (term j -> bucket j%8, sequential fp16 accumulate), then tree reduce
   ((0+1)+(2+3))+((4+5)+(6+7))."""
import os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); SIM=os.path.normpath(os.path.join(HERE,"..","sim"))
f16=np.float16
def h(v): return np.float16(v)
def pmac_ref(x, w):
    P=(x.astype(f16)*w.astype(f16)).astype(f16)
    buck=[f16(0)]*8
    for j in range(len(P)): buck[j%8]=f16(buck[j%8]+P[j])
    r=[f16(buck[0]+buck[1]),f16(buck[2]+buck[3]),f16(buck[4]+buck[5]),f16(buck[6]+buck[7])]
    r=[f16(r[0]+r[1]),f16(r[2]+r[3])]
    return f16(r[0]+r[1])
rng=np.random.default_rng(7)
K=64; L=80
X=np.zeros(K*L,dtype=f16); W=np.zeros(K*L,dtype=f16); R=np.zeros(K,dtype=f16)
for k in range(K):
    x=rng.uniform(-2,2,L).astype(f16); w=rng.uniform(-2,2,L).astype(f16)
    X[k*L:(k+1)*L]=x; W[k*L:(k+1)*L]=w; R[k]=pmac_ref(x,w)
def whex(fn,arr):
    with open(os.path.join(SIM,fn),"w") as f:
        for v in arr.view(np.uint16): f.write("%04x\n"%v)
whex("pm_x.hex",X); whex("pm_w.hex",W); whex("pm_ref.hex",R)
with open(os.path.join(SIM,"pm_dims.txt"),"w") as f: f.write(f"{K} {L}\n")
print(f"wrote {K} dot-products of length {L}")
