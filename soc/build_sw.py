#!/usr/bin/env python3
# build_sw.py - build the bare-metal Gemma3 inference ELF from gemma_soc.xsa.
# Run: vitis -s build_sw.py     (needs the Arm Cortex-A9 baremetal toolchain)

import vitis, os, glob, shutil

PROJ   = r"D:\Work\FPGA\gemma_pl\soc"
WS     = os.path.join(PROJ, "ws")
XSA    = os.path.join(PROJ, "gemma_soc.xsa")
SRC    = os.path.join(PROJ, "sw", "main.c")
VOCAB  = os.path.join(PROJ, "sw", "vocab.h")

PLAT   = "gemma_plat"
DOMAIN = "standalone_a9_0"
APP    = "gemma_app"

if os.path.isdir(WS):
    shutil.rmtree(WS, ignore_errors=True)

client = vitis.create_client()
client.set_workspace(WS)

platform = client.create_platform_component(name=PLAT, hw_design=XSA)
platform.add_domain(name=DOMAIN, cpu="ps7_cortexa9_0", os="standalone")
platform.build()
platform_xpfm = client.find_platform_in_repos(PLAT)

app = client.create_app_component(name=APP, platform=platform_xpfm,
                                  domain=DOMAIN, template="hello_world")
appsrc = os.path.join(WS, APP, "src")
shutil.copyfile(SRC,   os.path.join(appsrc, "helloworld.c"))
shutil.copyfile(VOCAB, os.path.join(appsrc, "vocab.h"))
app.build()

vitis.dispose()

elfs = glob.glob(os.path.join(WS, APP, "**", "*.elf"), recursive=True)
ps7  = glob.glob(os.path.join(WS, "**", "ps7_init.tcl"), recursive=True)
print("=== BUILD ARTIFACTS ===")
for e in elfs: print("ELF:", e)
for p in ps7:  print("PS7INIT:", p)
