# Qwen explicit expert arithmetic and short placement parity

The experimental Qwen builder accepts `ESE_QWEN4EXP_EXPERT_F32=1` and requests
the qualified original-weight/F32-input arithmetic for fused gate/up and indexed
down projections. This option does not open the production loader gate. Other
architectures and the default Qwen path retain their existing behavior. The
requested policy rejects unsupported types, biases, activations and adapters.

Evidence: `/home/xero/Desktop/test/results/ese-qwen-expert-f32-model-20261011`.
`qualification-current.json` records accepted checks and retained failures.
The model library under `ordered/libraries` is a separate diagnostic build:
only its loader-gate object enables the existing configure/runtime probe gate.
Saved compilation/link argv, object hashes and library hashes identify it.
Normal build libraries, model files and launcher configuration are unchanged.

## Why projection parity alone was insufficient

The first policy version executes matching gate/up and down arithmetic but
still changes model logits and routing when placement changes. Reproducible
traces locate the first CPU/cache difference at layer 0's weighted reduction:
maximum absolute difference `1.49e-8`, with its input projections identical.
The CPU reference permits fused multiplication/reduction while an adaptive
cache disables that existing experimental path. Hybrid execution additionally
reduces its two expert subsets separately before combining them.

The explicit policy now rounds routing-weight products to F32 separately. In
hybrid execution it retains outputs at their original router positions,
merges exclusive CPU/GPU positions, then adds in router order. This keeps
placement from changing the tested reduction arithmetic. The default path
retains its previous reduction. No tolerance was relaxed to obtain equality.

## Accepted checks

On the shared original coder IQ1_M artifact, nine teacher-forced token steps
produce 2,234,880 byte-identical logits across CPU experts, five GPU experts
with the remainder on CPU, and GPU cache execution. Every token's 48 MoE
routes and 12 QSA selections also match exactly. Probe callbacks verify the
requested precision and actual host/device execution; hybrid executes 48
gate/up and 48 down operations on each side per token. GPU cache execution
runs all 48 of each operation on GPU.

Serialized sequence snapshots match byte-for-byte across placements: the
eight-token prefix is 118,261,968 bytes and its next-token continuation is
118,289,636 bytes. This covers the state serialized by the sequence API,
including recurrent and attention state, rather than only final logits.

The compact-cache component regression rotates original expert rows to make
eight logical experts distinguishable, poisons host weights, requires leases,
delivers load completions in reverse order and forces physical-slot eviction.
Four original-payload format pairs cover all six actual expert quantizations.
All 24 native/all-API GPU runs pass on three GPUs, with 25,600 exact output
comparisons per run, no warm reacquisition, bounded cache allocation and final
lease drain. Twelve GPU memory runs report zero errors/leaks.

The default model control's 2,234,880 teacher values match the prior frozen
default build's first nine steps byte-for-byte. Eleven focused existing CTests
pass, including production loader rejection.

## Device isolation and memory checking

Initial full-model checks fail with 60 internal cuBLAS library-loading reports:
34 fatbinary errors, 20 missing GPU ISA 75 errors and six missing PTX errors.
The prior frozen ESE build reproduces the same reports. A cuBLAS-only wide
GEMM reproduces the loading errors despite analytically correct output.
Changing installed runtime selection or using a separately hash-verified
NVIDIA 13.2 sanitizer does not resolve them. Zero-byte cached sanitizer
launchers are rejected as invalid evidence, regardless of their exit status.

The minimal GEMM passes when only the actual Ampere target is visible. Final
full-model runs expose only the same physical RTX 3060 Ti/RTX 3080 pair,
remapping `-dev CUDA0,CUDA2` to `CUDA0,CUDA1`; the older backend-blocking shim
is removed. No hardware/budget or installed software changes are made.
Both hybrid/cache profiles pass the real system sanitizer with unchanged
`--report-api-errors all`, zero errors and zero leaks. Their logits, every
token's routes and both serialized sequence states match the same reference
exactly. RAM residency/staging caps and final lease/load/staging drain pass.
The three-visible-device failures remain recorded. This does not qualify a
full-model Turing run or erase those environment-dependent failures.

## Single-device 33-token memory checks

The RTX 3080 and RTX 3060 Ti each pass a separate original-weight, all-GPU
expert memory check with all CUDA API errors enabled and full leak checking.
Each executes 99 teacher-forced tokens across three 33-token reset/replay
rounds, including 66 exact replayed outputs. All 8,194,560 captured logits,
routes and input IDs match the corresponding same-device native control
byte-for-byte. Tracked native inventory is stable after warmup and expert/pool
leases drain. Both checks report zero errors and zero leaked bytes.

These profiles use F32 KV, a 512 MiB GPU expert cache and 64 MiB RAM expert
cache. GPU gate/up and down execution are verified for all 48 layers per
token. The sampled owned GPU peaks are 5,643,436,032 bytes on the RTX 3080
and 5,528,092,672 bytes on the RTX 3060 Ti, including instrumentation. Sampling
does not establish a complete physical memory cap or throughput result.

Evidence: `/home/xero/Desktop/test/results/ese-qwen-gpu-experts-memcheck-20261012`
and `/home/xero/Desktop/test/results/ese-qwen-experts-all-devices-memcheck-20261012/3060Ti-allGPU33-memcheck-audit.json`.
This result does not establish full-GPU Turing memory qualification or repair
the retained 33-token numerical-reference failures. The production gate stays
closed.

## Qualification limits

These are short real-artifact internal placement checks, not full upstream
model/quality parity, universal floating-point equality, long-context or
serving-stress qualification. Batched/prefill verification, residency-selected
placement, broader model families and complete resource ownership remain
unqualified by these checks. No matched Strata speed ratio or production
promotion is established.
