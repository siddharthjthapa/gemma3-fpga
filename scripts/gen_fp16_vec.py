#!/usr/bin/env python3
"""Generate random fp16 operand + reference vectors for tb_fp16 (mul/add)."""
import os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); SIM=os.path.normpath(os.path.join(HERE,"..","sim"))
os.makedirs(SIM, exist_ok=True)
rng=np.random.default_rng(1)
N=2000
# values in a safe normal range (avoid fp16 subnormals/inf; RTL flushes subnormals)
a=(rng.uniform(-8,8,N)).astype(np.float16)
b=(rng.uniform(-8,8,N)).astype(np.float16)
# keep magnitudes away from zero so exp!=0 (no subnormal operands)
a[np.abs(a)<np.float16(0.06)]=np.float16(0.5)
b[np.abs(b)<np.float16(0.06)]=np.float16(0.5)
mul=(a*b).astype(np.float16)
add=(a+b).astype(np.float16)
def whex(fn,arr):
    with open(os.path.join(SIM,fn),"w") as f:
        for v in arr.view(np.uint16): f.write("%04x\n"%v)
whex("fp16_a.hex",a); whex("fp16_b.hex",b)
whex("fp16_mul_ref.hex",mul); whex("fp16_add_ref.hex",add)
print(f"wrote {N} fp16 test vectors to {SIM}")
