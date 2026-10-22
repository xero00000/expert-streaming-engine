# CUDA device selection and fresh upload threads

The installed CUDA runtime can report device zero on a fresh host thread even
when that thread has no current CUDA context. ESE's device-selection helper
queried the device and returned early when the requested ordinal matched. An
unsuppressed all-API model check consequently reported nine handled
`CUDA_ERROR_INVALID_CONTEXT` diagnostics through `cudaGetDevice` and
`cuCtxGetDevice_v2`: one during memory planning and eight in model upload
workers. All 19 accepted-prefix comparisons passed, but the sanitizer returned
99. This is retained as failed qualification, rather than a clean memory run.

The native CUDA helper now always calls checked `cudaSetDevice`. HIP and MUSA
retain their existing query shortcut. No thread-local device cache is used:
driver clients can detach or change a context between backend calls. NVIDIA's
[runtime initialization documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#runtime-initialization)
states that `cudaSetDevice` initializes the runtime and primary context from
CUDA 12 onward. This qualification uses the locally installed CUDA runtime;
it does not establish behavior on every older runtime or on HIP/MUSA.

Evidence is in
`/home/xero/Desktop/test/results/ese-cuda-device-context-20261011`.
The source, binary, library hashes, commands, logs and original failed controls
are preserved independently of the prior immutable checkpoint generations.

- The old helper fails the fresh-thread/context-detachment checks with 33
  missing contexts under isolated device visibility and 49 with all three
  devices visible. Unsuppressed sanitizer controls report 33 and 73 API errors.
- The candidate passes on each of the RTX 3060 Ti, RTX 2080 SUPER and RTX 3080,
  and with all three visible together. All eight native/sanitizer runs exit
  zero, with no missing contexts, wrong device selections or sanitizer errors.
- `test-cuda-device-context` checks the actual backend helper on the main
  thread, eight fresh workers, device switches, explicit driver detachment and
  repeated selection. It is registered only in CUDA builds.
- Nine normal CUDA-build regressions pass, including architecture rejection,
  checkpoint isolation, complete prefix states, PLE journals and operator
  checks. The context probe is actual CUDA execution; the other default
  regressions do not imply a new complete all-device kernel qualification.

One-million-call native screens measure about 28 ns per warm call for the old
query shortcut and 47–49 ns for explicit selection. These are single-process
component observations, not a whole-model speed claim.

The first diagnostic incorrectly timed the old helper while the main thread
still had no context, repeating the failure in the timing loop. That owned
diagnostic was terminated, its failure retained, and the revised timing
initializes a context before measuring. Its error count is not the model's
nine-error count and is not accepted timing evidence.

The full-model unsuppressed sanitizer rerun is pending under
`run-model-qualification.py`, followed sequentially by chunk-abort checks,
ordinary teacher controls and the independent-sequence bank candidate.
The wider upstream/hybrid correctness, actual MTP, long-context/resource,
matched-performance and release gates remain open.
