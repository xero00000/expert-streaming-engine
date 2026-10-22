# CUDA startup and reduction lifecycle validation — 2026-10-13

## Reproduced failures and repairs

On the three-card Ubuntu 24.04 test system, device enumeration alone created
seven NCCL ranks and held 792 MiB reported by NVML (346/232/214 MiB). No model
or backend had been created. Discovery now initializes no NCCL ranks and reports
no owned GPU memory rows. Communicators are created once at the first eligible
all-device reduction; unused pair communicators are no longer created.

Initialization and grouped calls share a mutex. A failed initialization cleans
up its ranks and retains the existing reduction fallback without repeated
allocation attempts. A two-participant reduction with three visible cards
previously asserted; it now uses that fallback. This follows
[NVIDIA's communicator thread-safety rules](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html).

Single-device model planning also queried free memory on other selected-list
rows, creating unselected CUDA primary contexts. Both native queries now use
only the main row for NONE mode. Deterministic cumulative split points retain
list cardinality without querying unused cards or dividing by zero free memory.
The permanent fresh-process primary-context regression fails against the
archived previous native library and passes after the repair.

The new reduction memory check found two 1024-byte scratch allocations left
behind by backend destruction. The destructor now frees its owned copy buffer.
The failed run and its previous library are retained; the corrected registered
memory check reports zero errors and zero leaked bytes.

## Evidence

Artifacts: `/home/xero/Desktop/test/results/ese-cuda-startup-memory-20261013`.

- F32, F16 and BF16 exact integer-sum controls: two contexts, eight repetitions
  each, on two and three visible GPUs. Concurrent first use creates one clique;
  grouped rank calls remain serialized. All created ranks are destroyed once.
- The two-participant/three-visible fallback creates no NCCL ranks and preserves
  exact sums. Injected initialization failure aborts the created ranks once,
  makes no collective calls, and preserves exact fallback output.
- 12 CUDA, six CPU and six strict CPU ASan/UBSan/leak regressions pass. The
  optional registered NCCL lifecycle memcheck passes on all three visible GPUs.
- 13 final release servers retain all 32 saved probabilities and content exactly
  against the previous qualified release build. Twelve real-plan cases preserve
  exact output through 512→256→512. Strict sanitizer HTTP resize uses its own
  build teacher and is exact through the same transitions.
- Actual single-device GPU server rows are restricted to the requested physical
  card on this hardware, including reordered device lists and CUDA visibility.

## Limits

CUDA-host pinned allocations can still create a driver context on a CPU launch
of a CUDA build. Fixed zero-layer and peer-capable hardware allocation behavior
are not qualified as exclusive device ownership. Complete vendor/driver memory
admission and CUDA reduction scratch budget accounting remain unqualified.

These checks use generated small models and synthetic reductions. Arithmetic
kernels, external references, tolerances, weights and the normal Qwen gate are
unchanged; subset dispatch and accumulation types are tested as described.
Original Qwen numerical/full-model/64K, concurrent serving and matched performance
gates remain open. Existing installer candidates predate these changes.
