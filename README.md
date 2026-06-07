# Gemma3 in FPGA fabric

A **600K-class, scaled-down Gemma3-style transformer** whose **entire forward pass
runs in fp16 SystemVerilog RTL** on the programmable logic of a Xilinx Zynq-7010
(Z-turn Lite board). This is not the full Gemma3 model; it is a small from-scratch
model with Gemma3 architecture conventions (see the training section below).
The ARM Cortex-A9 (PS) only feeds tokens,
samples (greedy argmax), and prints over UART, while every layer of the model
(Gemma RMSNorm, per-head QK-norm,
split-half RoPE, grouped-query sliding-window attention with an on-chip fp16 KV
cache, the GELU-gated FFN, the post-sublayer norms, and the tied classifier)
executes in the FPGA at 100 MHz.

The model is a 600K-class Gemma3 checkpoint (dim 80, hidden 240, 7 layers, 4 query
heads / 1 KV head, head_dim 20, vocab 1024, sliding window 64). Weights are fp16,
and the datapath computes in fp16 throughout. On hardware it generates a coherent
short story and reports its own throughput:

```
=== Gemma3-600K transformer on Z-turn PL fabric @100MHz ===
The little robot lived in the red shed. Every morning it rolled across the floor.
One day Arun brought a small bell to the robot and said, "Can you help me?" ...
"small work can solve a big worry." That evening everyone remembered the small
good work.
=== 128 tokens in 1154034 us  (1st tok 6698 us) ===
=== throughput = 110.9 tokens/sec ===
```

## Numeric fidelity

The whole datapath is fp16: weights and the KV cache are stored fp16, and every
multiply, add, dot-product accumulation, exp, and reciprocal-sqrt runs in fp16.
The core is therefore **argmax-faithful, not bit-exact** to a sequential fp32 host
run: the fp16 reduction order and the fp16 transcendental approximations differ
slightly from the reference, but the chosen tokens match and the generated text is
the same coherent story produced by the golden model. A multi-token simulation
(`sim/tb_gen.sv`) and an on-board self-test both reproduce the golden token stream.

## The model and how it was trained

This is not a distilled or pruned copy of Gemma3-270M. It is a 600K-class model
**trained from scratch** with Gemma3-style architecture conventions, so it fits on the FPGA. The trained weights (`golden/gemma3_600K.bin`,
599,720 fp16 values), the tokenizer (`golden/gemma3_tok1024.bin`), and the C
reference runtime (`golden/run_ref.c`) are shipped here;

Key choices:

- **Custom 1024-token tokenizer.** The official Gemma3 tokenizer has a 262K-token
  vocabulary, whose embedding table alone would blow the entire weight budget. The
  model instead uses a compact byte-BPE tokenizer with 1024 tokens (max piece
  length 24) and Gemma-compatible special IDs: `0 = <pad>`, `1 = <eos>`,
  `2 = <bos>`, `3 = <unk>`, `4..259` = raw byte tokens, `260..1023` = learned BPE
  pieces.
- **Narrow synthetic corpus.** Training data is a generated story corpus of about
  60,000 stories rather than TinyStories or web text, because sub-million-parameter
  models learn a clean constrained distribution far better than broad text. The
  generator binds each object to a matching problem, fix, and result for
  consistency, for example `blue box -> would not open -> found the small silver
  key -> opened with a click`. This is also why correct continuations often repeat
  an object name, and why broad repetition penalties were deliberately not used.
- **Size tradeoff.** `dim=80, hidden_dim=240, layers=7, vocab=1024` was chosen to
  keep the larger vocabulary and push depth past 6 layers while staying just under
  the 600K cap (599,720 weights).

Last completed training run (PyTorch, CUDA):

```text
tokenizer_vocab=1024  max_piece_len=24
weights=599720
step 2500/2500  train_loss=0.3678  val_loss=0.3722
  (2500 steps, batch 160, seq-len 128)
```


## Differences from the llama2 build


### Model architecture (Gemma3 vs llama2)

| aspect | llama2-260K | Gemma3-600K |
|---|---|---|
| dimensions | dim 64, hidden 172, 5 layers | dim 80, hidden 240, 7 layers |
| heads | 8 query / 4 KV (head_size 8) | 4 query / 1 KV (head_dim 20), GQA group of 4 |
| RMSNorm | `out = w * (ss * x)`, eps 1e-5 | Gemma form `out = x * ss * (1 + w)`, eps 1e-6 |
| QK norm | none | per-head RMSNorm on q and k before RoPE |
| RoPE | interleaved pairs `(x[2i], x[2i+1])` | split-half `(x[i], x[i+half])` (NeoX style), theta 10000 |
| attention | full causal | sliding window of 64 positions |
| FFN | SwiGLU (SiLU gate) | GeGLU (GELU-tanh gate) |
| sublayer norms | pre-norm only | pre-norm plus a post-attention and a post-FFN RMSNorm before each residual add |
| embedding | used as-is | scaled by sqrt(dim) |
| classifier | tied to the embedding table | tied to the embedding table (same) |
| weights | fp32 | fp16 |

### Hardware implementation (this core vs the fp32 llama core)

| aspect | llama_pl (fp32) | gemma_pl (fp16) |
|---|---|---|
| arithmetic | `fp32_mul_p / fp32_add_p / fp32_exp_p / fp32_rsqrt_p`, fp32 `pmac` | `fp16_*` equivalents, fp16 `pmac16` (8 round-robin fp16 accumulators plus a tree reduce) |
| weight beat | 64-bit AXI beat carries 2 fp32 | 64-bit AXI beat carries 4 fp16 |
| matmul lanes | 2-lane row-interleaved (2 output rows per beat) | 4-lane row-interleaved (4 output rows per beat) |
| KV cache | fp32 | fp16 (half the BRAM); per-entry stride padded to a power of 2 so the cache address is a shift, not a DSP multiply |
| nonlinearities | exp plus rsqrt; SiLU via reciprocal | exp plus rsqrt; GELU computed as `x / (1 + e^(-2z))` so no tanh unit and no divide unit are needed (every reciprocal is `rsqrt(d)^2`) |
| RoPE tables | streamed from DDR | streamed from DDR, with the per-position stride padded to a multiple of 4 so each read lands on an 8-byte AXI boundary (see note below) |
| timing | fp32 paths registered to close 100 MHz | act-RAM reads registered and KV writes registered to close 100 MHz in fp16 |
| resources | about 10.9K LUT, 15 DSP, 33 BRAM | about 7.8K LUT, 14 DSP, 32.5 BRAM |

A subtle hardware-only bug worth recording: the real PS7 HP port aligns an AXI read
address down to the transfer-size (8-byte) boundary. The RoPE table originally had
a stride of 10 (`head_dim/2`), so odd positions produced a 4-byte-aligned address
that the hardware silently shifted, corrupting the rotation on every other token.
An idealized DDR model does not show this. The fix pads the RoPE table stride to a
multiple of 4, and `rtl/ddr_model16_rl.sv` now models the alignment behaviour (plus
AR/R latency and rvalid gaps) so simulation matches the board.

## Repository layout

```
rtl/        the accelerator: design SystemVerilog plus the AXI DDR sim models
sim/        testbenches plus golden vectors plus the fp16 weight hex (model.hex)
scripts/    golden model, weight-blob and vector generators, the xsim runner
soc/        Zynq SoC wrapper, build/flash scripts, bare-metal PS app
golden/     model checkpoint, tokenizer, the C reference (run_ref.c)
prebuilt/   ready-to-flash bitstream and hardware handoff (.bit / .xsa)
```

### rtl/ (the core)

| file | role |
|---|---|
| `gemma_fwd_p.sv` | the 100 MHz pipelined fp16 forward pass (main design) |
| `gemma_top.sv` | AXI-Lite control slave plus weight-read master plus the core |
| `pmac16.sv` | interleaved-accumulator fp16 dot-product engine |
| `axi_rd_m16.sv` | 64-bit AXI4 read master, fp16 payload (4 fp16 per beat, wide and narrow taps, 4 KB-safe bursts) |
| `fp16_mul_p / fp16_add_p / fp16_exp_p / fp16_rsqrt_p .sv` | pipelined fp16 primitives |
| `fp2int_round16.sv`, `int2fp16.sv` | fp16-to-int and int-to-fp16 helpers used by exp |
| `ddr_model16.sv` | idealized AXI DDR model for fast simulation |
| `ddr_model16_rl.sv` | hardware-faithful AXI DDR model (latency, rvalid gaps, 8-byte address alignment) |

### sim/ (testbenches and their data)

| file | role |
|---|---|
| `tb_gemma.sv` | full forward pass, checks argmax against the fp16 golden logits |
| `tb_gemma_rl.sv` | full forward pass against the hardware-faithful DDR model |
| `tb_top.sv` | full SoC path: AXI-Lite control plus logit BRAM readback |
| `tb_gen.sv` | multi-token generation, decodes each token to text in the console |
| `tb_fp16.sv` | `fp16_mul_p` / `fp16_add_p` unit test |
| `tb_pmac16.sv` | `pmac16` dot-product unit test |
| `tb_trans16.sv` | `fp16_exp_p` / `fp16_rsqrt_p` unit test |
| `model.hex` | the fp16 weight blob loaded by the full-forward testbenches |
| `*.hex` (others) | golden reference vectors loaded by the testbenches |

The testbenches load their data with bare filenames, so run the simulator from
inside `sim/` (the runner script below does this).

### scripts/

| file | role |
|---|---|
| `golden.py` | NumPy reference that ports `run.c`; generates text and dumps op-level golden vectors |
| `prec_probe.py` | fp32-vs-fp16 viability study; emits the fp16 golden logits and op dumps for the RTL testbenches |
| `gen_blob.py` | builds the fp16 weight blob (`sim/model.hex` and `soc/model.bin`) with the RoPE tables baked in and the matmul matrices 4-row interleaved |
| `gen_fp16_vec.py`, `gen_pmac16_vec.py`, `gen_trans16_vec.py` | regenerate the unit-test golden vectors |
| `gen_sim_gen.py` | tokenizer tables plus prompt tokens for `tb_gen.sv` |
| `sim.ps1` | compile and run an xsim testbench (runs from `sim/`) |

### soc/

| file | role |
|---|---|
| `build_gemma.tcl` | Vivado: PS7 plus accelerator plus interconnect to bitstream plus XSA |
| `gen_assets.py` | tokenizer to the PS `vocab.h` and the prompt tokens (the weight `model.bin` comes from `gen_blob.py`) |
| `build_sw.py` | Vitis: platform plus bare-metal app to `gemma_app.elf` |
| `flash_gemma.tcl` | xsdb: program PL, load weights to DDR, run the app |
| `flash_gemma_slow.tcl` | same flash but at half clock (50 MHz) for timing diagnosis |
| `gemma_top_v.v` | thin Verilog wrapper so the block design can reference `gemma_top` |
| `run.ps1` | one-command build/flash wrapper |
| `sw/main.c` | bare-metal inference loop (token in, fp16 logits, argmax, decode, print) |

## Hardware

- Board: Z-turn Lite (Xilinx Zynq xc7z010clg400-1)
- Tools: Vivado / Vitis 2025.1
- PL utilization: about 7.8K LUT (44%), 5.4K FF (15%), 32.5 BRAM tiles (54%), 14 DSP (17.5%), timing closed at 100 MHz
- UART: J24 (PS UART1), 115200 8N1

## Build and run

From `soc/`, with Vivado/Vitis 2025.1 on the PATH:

```powershell
.\run.ps1 -All     # bitstream (Vivado) + model.bin/vocab.h + ELF (Vitis), then flash
.\run.ps1 -Build   # rebuild app + assets, reuse the bitstream, then flash
.\run.ps1          # flash existing artifacts over JTAG
```

Then open J24 (PS UART1) in a terminal at 115200 8N1 to watch the story generate.

Manual equivalents:

```powershell
# 1. build the fp16 weight blob (golden/*.bin -> sim/model.hex and soc/model.bin)
python scripts\gen_blob.py

# 2. hardware
vivado -mode batch -source soc\build_gemma.tcl     # -> gemma_soc.bit + gemma_soc.xsa

# 3. PS assets (tokenizer -> soc/sw/vocab.h and prompt tokens)
python soc\gen_assets.py

# 4. application
vitis -s soc\build_sw.py                            # -> ws\gemma_app\build\gemma_app.elf

# 5. flash
xsdb soc\flash_gemma.tcl <ps7_init.tcl> gemma_soc.bit model.bin <gemma_app.elf>
```

To run the prebuilt design, use the `.bit` / `.xsa` in `prebuilt/` together with
`soc/model.bin` (regenerate it with `gen_blob.py`) and a freshly built ELF.

## Simulate

The full-forward testbenches read the fp16 weight blob, so generate it first:

```powershell
python scripts\gen_blob.py         # produces sim/model.hex
```

Then run a full forward-pass check (any SystemVerilog simulator; xsim shown). The
helper script compiles the design from `..\rtl` together with the chosen
testbench and runs it from inside `sim/`:

```powershell
.\scripts\sim.ps1 -Top tb_gemma -Files "fp16_mul_p.sv,fp16_add_p.sv,fp2int_round16.sv,int2fp16.sv,fp16_exp_p.sv,fp16_rsqrt_p.sv,pmac16.sv,axi_rd_m16.sv,ddr_model16.sv,gemma_fwd_p.sv" -Tb "tb_gemma.sv"
# expects RESULT: PASS (argmax)
```

`tb_gemma_rl.sv` runs the same check against the hardware-faithful DDR model
(swap `ddr_model16.sv` for `ddr_model16_rl.sv`), `tb_top.sv` exercises the full SoC
AXI-Lite path, and `tb_gen.sv` generates a multi-token story to the console. The
`tb_fp16 / tb_pmac16 / tb_trans16` benches check the fp16 primitives against the
NumPy vectors shipped in `sim/`.

## Notes and limits

- Sequence positions are capped at 128: the on-chip fp16 KV cache is 128 deep per
  layer so it fits in xc7z010 BRAM. Positions must be issued in order (0, 1, 2, ...)
  because the KV cache persists in the PL across tokens. The sliding window is 64.
- Sampling is greedy/argmax (deterministic). Temperature sampling could be added in
  `sw/main.c`.
- `model.bin` is the fp16 weight blob loaded straight to physical DDR over JTAG; the
  PL reads it through the HP0 port while the PS drives control over AXI-Lite.
- All AXI read base addresses must be 8-byte aligned; the weight-blob generator pads
  the RoPE table stride to keep every per-position read aligned.

## License

MIT, see [LICENSE](LICENSE).
