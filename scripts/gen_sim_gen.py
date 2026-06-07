#!/usr/bin/env python3
"""Emit sim-side tokenizer data for tb_gen (multi-token generation testbench):
   sim/vocab_off.hex   (vocab+1) u32 byte offsets into the piece blob
   sim/vocab_bytes.hex  raw piece bytes, one hex byte per line
   sim/prompt_toks.hex  greedy-encoded prompt token ids
   sim/gen_meta.svh     localparams (NPROMPT, NOFF, NBYTES)
"""
import os, struct
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.normpath(os.path.join(HERE,".."))
GOLD=os.path.join(ROOT,"golden"); SIM=os.path.join(ROOT,"sim")
raw=open(os.path.join(GOLD,"gemma3_tok1024.bin"),"rb").read()
(vocab,)=struct.unpack("<I",raw[:4])
offs=list(struct.unpack("<%dI"%(vocab+1),raw[4:4+4*(vocab+1)]))
body=raw[4+4*(vocab+1):]
pieces=[body[offs[i]:offs[i+1]] for i in range(vocab)]
maxlen=max(len(p) for p in pieces)

def encode(text,bos=True):
    b=text.encode(); toks=[2] if bos else []; pos=0
    while pos<len(b):
        bid,bl=-1,0
        for i in range(4,vocab):
            p=pieces[i]
            if len(p)>bl and b[pos:pos+len(p)]==p:
                bid,bl=i,len(p)
                if bl==maxlen: break
        if bid<0: bid,bl=3,1
        toks.append(bid); pos+=bl
    return toks

PROMPT="The little robot lived in the red shed. Every morning it rolled across the floor. One day "
ptoks=encode(PROMPT)
with open(os.path.join(SIM,"vocab_off.hex"),"w") as f:
    f.write("\n".join("%x"%o for o in offs)+"\n")
with open(os.path.join(SIM,"vocab_bytes.hex"),"w") as f:
    f.write("\n".join("%02x"%c for c in body)+"\n")
with open(os.path.join(SIM,"prompt_toks.hex"),"w") as f:
    f.write("\n".join("%x"%t for t in ptoks)+"\n")
with open(os.path.join(SIM,"gen_meta.svh"),"w") as f:
    f.write(f"localparam int NPROMPT={len(ptoks)};\n")
    f.write(f"localparam int NOFF={vocab+1};\n")
    f.write(f"localparam int NBYTES={len(body)};\n")
print(f"prompt={len(ptoks)} toks {ptoks}; vocab={vocab}; bytes={len(body)}")
