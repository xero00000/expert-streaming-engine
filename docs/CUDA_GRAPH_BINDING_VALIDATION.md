# CUDA graph backing-allocation identity

Cached CUDA executables now track the identities of their ordinary CUDA backing
buffers, in addition to tensor addresses, shapes and operation parameters.
Every CUDA buffer receives a monotonic allocation ID in its `.cu`-local private
context. Replacing a buffer recreates the executable, including when allocation
addresses and graph-arena addresses are recycled. Persistent allocations retain
ordinary graph replay. Public buffer and private backend-context layouts are
unchanged; the internal CUDA buffer context is owned entirely by this source.

Binding records store distinct buffer IDs per cached graph. They are removed
when that graph is evicted or its backend is destroyed. The existing graph
cache eviction policy remains in effect. IDs cover ordinary CUDA buffers;
special/imported buffer classes and broader lifetime/concurrency cases require
their own qualification. Binding metadata and scan cost are separate from
GPU payload/scratch budgets and still need performance/resource accounting.

Evidence: `/home/xero/Desktop/test/results/ese-cuda-graph-binding-20261011`.
The identical pre-fix binary/library control exits11 under all-API memcheck.
Its core shows SIGSEGV in the driver/sanitizer call path at `cuGraphLaunch`;
no memory-access errors were reported before termination. An eight-case control
fails with recycled graph metadata and passes with its32arenas retained.
The earlier unlimited retention attempt hits GGML's64-context limit and is not
counted as a passing qualification.

The repaired library passes540PLE prefix cases on each GPU natively and with
unsuppressed all-API memory checks, all six exits0 and zero errors. Tests retain
their ordinary free/reallocate behavior and two executions per graph. The
metadata-retention diagnostic is disabled. This is graph-enabled-backend
coverage; it includes actual captured execution and existing compatibility
fallbacks, rather than asserting that every case is captured.

Real-model qualification uses seeded target sampling at prefixes8/32 with
CPU/GPU-fallback checkpoints, depths2/4/8 and every rejection/full-accept count,
plus ordinary teacher/lifecycle comparisons. All 56 seeded cases pass exactly,
including sampler decisions/state, sequence state, logits and applicable future
draws. Both ordinary teacher/lifecycle runs pass all 21 checks; their 8,194,560
F32 teacher logits per stream, baseline state and routes remain byte-identical.
Fourteen refreshed normal CTests and two refreshed ASan/UBSan/leak checks pass.
These use controlled expert arithmetic, not a production hybrid or upstream
full-model oracle. The production architecture gate and launchers remain unchanged.
