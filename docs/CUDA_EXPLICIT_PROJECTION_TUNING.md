# Explicit CUDA projection tuning

October 10, 2026. Experimental, disabled by default.

`ESE_CUDA_EXPLICIT_AUTOTUNE=1` measures the existing small explicit-F32 projection kernel on the GPU and tensor shapes encountered by a loaded model. It chooses among 1, 2, 4 and 8 output warps per block. This first tuning component uses compiled kernel variants; it does not add a runtime compiler dependency or compile an entire model.

Each output retains its original 32-lane products, double accumulation and shuffle reduction. Calibration compares four input patterns bitwise with the four-warp reference before considering a candidate. Five graph-timed samples provide a median; selection requires more than 5% measured improvement. Output arithmetic, routing, quantization and CPU/GPU placement remain fixed.

Calibration happens before inference graph capture, using separate input/output scratch. Leaf, contiguous F32/F16/BF16 CUDA weights, one input vector, widths 1024–65536 and at most 256 output rows are eligible. Split/host weights, computed weights, strided inputs and batched inputs use the existing path. The scalar control takes precedence. Plans are immutable after admission, keyed by device/type/width/rows and capped at 128 per process. A new process recalibrates for its actual backend and driver; there is no persistent plan cache. Transient scratch is at most 263,168 bytes, then freed; allocation failure retains the reference choice. Shapes first encountered while another backend captures a graph retain the reference to avoid changing an existing executable's plan.

## Validation

Evidence: `/home/xero/Desktop/test/results/ese-model-plan-20261010`.

- F32, F16 and BF16 projections, three geometries, one/three input vectors, four input patterns, changed weights and repeated execution match the frozen backend exactly on CUDA0/1/2. Both graph and non-graph runs pass.
- `test-explicit-projection` compares outputs bitwise with an independent CPU reproduction of the existing lane arithmetic. The CUDA test enables tuning; it covers all three local GPUs.
- The 16-token fixed-placement production-artifact probe matches 3,973,120 logits and 60 canonical route tensors exactly. This establishes candidate/control equivalence for that sequence, not complete model/reference parity.
- Focused CUDA-build CTests pass 12/12, including the production rejection control, Qwen operators/storage/schema, expert cache, KDA and the new projection regression.
- Separate unsuppressed memory/API checks qualify the small projection and generated-kernel families. The broader prepared-weight pilot has older-driver cuBLASLt fatbinary-load failures on CUDA0/2; do not treat that pilot as sanitizer-clean.

## Timing and scope

Same fixed five-GPU/five-CPU expert placement, 64 MiB expert RAM/4 MiB staging and 2680 MiB expert VRAM per GPU on CUDA0+CUDA2. Three repeats per workload; identical output and arithmetic-quality response.

| Workload | Control tok/s | Tuned tok/s | Change |
|---|---:|---:|---:|
| Hi, 32 generated tokens | 4.219 | 4.331 | +2.65% |
| Log prompt, 256 input/16 generated tokens | 3.371 | 3.284 | −2.60% |

Mode order was sequential; OS cache, clocks and background activity were uncontrolled. These mixed short-run results do not establish a general inference speedup. Defaults remain unchanged. Both runs have identical 268,326 RAM misses, 183,746,124,800 logical source bytes and 7,869,132,800 GPU upload bytes; cache/staging caps and clean shutdown checks pass. Storage waits and CPU expert work still dominate this constrained configuration.

An independent NVRTC pilot compiled shape-specific native CUBINs and reproduced existing outputs on all three GPUs. The installed driver rejected CUDA 13.4-generated PTX; native sm_75/sm_86 CUBINs worked. That pilot is an architectural experiment, separate from the installed tuner. Preconverting hyper-connection weights also improves isolated projections, but full preparation needs about 2.37 GiB extra; there is no unbudgeted prepared-weight cache in the engine.
