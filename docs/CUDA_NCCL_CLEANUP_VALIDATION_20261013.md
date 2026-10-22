# CUDA NCCL communicator cleanup validation

Date: 2026-10-13. This qualifies communicator ownership and shutdown in the
tested Ubuntu CUDA runtime. Release-wide readiness remains incomplete.

## Failure and repair

The Ubuntu build discovers NCCL and creates communicators for every visible
GPU. With three GPUs, it creates three main handles and two pairs of two
handles. These handles previously had no cleanup owner. Compute Sanitizer
reported **139,466,068 bytes leaked in 119 allocations**, with exit code 99.
Each single-visible-GPU control passed because it creates no communicator.

`ggml_cuda_info()` now registers a separate lifetime owner after initializing
its static device information. The owner destroys every nonnull communicator
in a grouped teardown. It runs before the device information and CUDA runtime
are destroyed. The object returned by `ggml_cuda_init()` does not own cleanup,
so copying or moving that initialization result cannot release live handles.

Failed initialization releases allocated handles through `ncclCommAbort()`.
The existing fatal policy for a failed pair initialization is retained, with
cleanup performed before the abort. An asynchronous communicator error also
selects abort cleanup. Cleanup errors are reported directly to stderr.

The implementation uses the existing NCCL destroy/abort APIs. It does not
require an explicit `ncclCommFinalize()` call. NVIDIA's
[communicator API documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html)
and the pinned
[NCCL 2.25.1 implementation](https://github.com/NVIDIA/nccl/blob/v2.25.1-1/src/init.cc)
were inspected for teardown and grouping behavior. Execution qualification
uses NCCL 2.25.1; other NCCL versions are not execution-qualified here.

## Environment and results

The isolated Ubuntu 24.04 environment uses CUDA 12.8.61, NCCL
`2.25.1-1+cuda12.8`, the Ubuntu GCC 13 toolchain, Release builds, and CUDA
architectures `75-real;86-real;86-virtual`. The normal Qwen gate remains OFF.
No model arithmetic, reference, tolerance, or weight was changed.

| Check | Result |
| --- | --- |
| Complete native CPU build and suite | Build passes; 91/92 tests pass |
| Complete native CUDA build and suite | Build passes; 96/97 tests pass |
| New registered shutdown memory regression | Passes, including process teardown |
| RTX 3080, RTX 3060 Ti, RTX 2080 SUPER, all GPUs visible | Each passes with zero errors and zero leaked allocations |
| Each GPU exposed alone by UUID | Each passes with zero errors and zero leaked allocations |
| Two GPUs exposed by UUID | Passes with zero errors and zero leaked allocations |
| Three-GPU communicator ownership audit | All seven handles destroyed exactly once |
| Two-GPU communicator ownership audit | Both handles destroyed exactly once |
| Injected main initialization failure | All three allocated handles aborted; workload and memory check pass |
| Injected first/second pair initialization failure | All five/seven allocated handles aborted before the expected fatal exit |

The only full-suite failure is the existing `test-deepseek-v41-block` BF16
reference comparison. It remains a release blocker. Primitive-control reports
on all three GPUs are byte for byte identical to their pre-fix reports.
The initialization fault controls inject an error after NCCL has successfully
allocated real handles; they qualify the cleanup branches, not a hardware
failure recovery claim. Four-GPU layouts and full-model NCCL workloads are
not qualified by this evidence.

## Permanent regression

Enable the optional hardware check in a CUDA test build:

```sh
cmake -S . -B build-cuda-checks -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_CUDA_MEMCHECK_TESTS=ON -DESE_QWEN4EXP_PROBE_GATE=OFF
cmake --build build-cuda-checks --parallel 6
ctest --test-dir build-cuda-checks -R '^test-cuda-shutdown-memcheck$' \
  --output-on-failure
```

The registered test runs the original DeepSeek primitive fixture on CUDA
device 0 under `compute-sanitizer --tool memcheck --leak-check full
--error-exitcode 99`. It runs serially and includes shutdown. A CPU-only
configuration with this option enabled fails explicitly. To cover the NCCL
path, use a build with NCCL enabled and expose at least two GPUs. The option is
OFF by default because it requires GPU hardware and Compute Sanitizer.

## Evidence and reproducibility

Evidence is preserved under `results/ese-nccl-cleanup-20261013`, including
commands, process records, terminal exits, JUnit reports, source before/after,
ownership audits, the scoped patch, and a SHA-256 manifest.

The earlier Ubuntu archive froze executables without their external build-tree
shared libraries. The initial pre-rebuild negative used those still-original
mutable libraries; its loader report records that limitation. For an
independent retained negative, the original CUDA translation unit was then
rebuilt and linked separately with 261 verified unchanged object inputs.
Its retained library reproduces the same 119-allocation leak, with loader
resolution explicitly checked. The new frozen runtime includes executables
and all three project shared libraries together; its loader and memory check
also pass. The earlier sealed evidence is unchanged.

GPU tests cover all three actual cards and use UUIDs for single/two-GPU
visibility. CUDA's device enumeration differs from the NVML index order in
this environment; this validation does not equate those indices.
