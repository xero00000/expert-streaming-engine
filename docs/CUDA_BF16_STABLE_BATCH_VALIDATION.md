# Experimental scalar-equivalent BF16 projection batches

`ESE_CUDA_BF16_STABLE_BATCH=1` enables an experimental CUDA path for contiguous,
unbiased DEFAULT BF16 weights with F32 input/output, width2560, rows2560/10240
and2..8queries. Other shapes, explicit-F32 precision, padded tensors and split
buffers retain existing dispatch. The default remains disabled.

## Implementation

The backend asks cuBLASLt for its scalar algorithm configuration, then checks
that configuration against the actual batch layout. It first checks a dense
matrix and then a strided batch with broadcast weights. It does not hard-code
algorithm IDs or select a configuration by looking at the output values.

Plans are prepared outside CUDA graph capture and owned by each CUDA backend
context. The cache has at most32entries, including unsupported entries; cleanup
releases layouts, preferences and vendor handles when the backend is freed.
Actual requested scratch (BF16 input/output plus vendor workspace) is capped
at512KiB. Allocator reservation and vendor initialization costs are separate.
Algorithm search uses the same8MiB preference ceiling as the standalone study.

One input conversion, one batched matmul and one output conversion replace the
per-query conversion/matmul loop for qualified shapes. Single-query dispatch
remains the reference. This path runs before the existing dense-vector diagnostic;
other projection formats in that diagnostic still use its old loop.

## Evidence and limits

Standalone evidence: `/home/xero/Desktop/test/results/ese-bf16-lt-20261010`.
Actual layer1 key/value weights are from the read-only shared Qwen artifact.
Fixed scalar-configuration reuse passes2048cases per GPU on all three devices,
including query counts1..8,32input patterns, tight/padded rows and weight offsets
0/128. GPU F32-to-BF16 and BF16-to-F32 conversion plus captured CUDA graph replay
also passes2048cases per device, every F32 output bit matching the scalar path.

All-API memory checks pass with zero errors under per-device UUID isolation.
The original all-visible Ampere heuristic run reports48driver API errors while
loading fatbinaries for ISA75; it is retained as a failed control. No invalid
buffer access was reported there. Mixed-device visibility remains a separate
qualification concern; no API-error reporting was suppressed.

Warm eight-query pipeline medians, seven alternating100-iteration repeats:

| GPU | Rows | Eight scalar calls | Parallel pipeline | Ratio |
|---|---:|---:|---:|---:|
| RTX3060Ti |2560|315.84us|42.04us|7.51x|
| RTX3060Ti |10240|1039.97us|992.18us|1.05x|
| RTX2080SUPER |2560|276.83us|231.42us|1.20x|
| RTX2080SUPER |10240|961.80us|910.25us|1.06x|
| RTX3080 |2560|217.24us|28.99us|7.49x|
| RTX3080 |10240|711.38us|91.41us|7.78x|

These are resident-weight projection pipelines. They exclude algorithm/graph
preparation, storage, expert loading, routing and CPU expert computation; clocks,
OS and background graphics were uncontrolled. They are not whole-model or
accepted-token throughput measurements.

Backend evidence: `/home/xero/Desktop/test/results/ese-bf16-lt-backend-20261011`.
The initial missing `cstring` compile attempt is retained separately from the
successful build. The native backend test passes64cases per GPU against actual
weights, every scalar bit and a quantization-aware dequantized double oracle,
including repeated graph execution and changing inputs. Trace logs confirm
seven batch plans per matrix and actual use of the new path.

The first integrated all-API run exposed handled CUDA graph-update failures
when a reused graph arena changed node count. The backend now recreates its
executable on that structural change. With this fix, all three GPUs pass the
64-case native test and unsuppressed all-API memory checks under per-device
UUID isolation, with zero reported errors. The failing control remains archived.

Both immutable backend generations completed the280-case real-model
sampler/state matrix: CPU/GPU-fallback checkpoint modes, prefixes8/32,
depths2/4/8, every output/rejection position and five sampling policies.
Verification decisions, recovered sequence bytes, logits, sampler/RNG/adaptive
state and applicable future draws match their scalar teachers exactly. Expert
RAM/staging limits and final drain pass. These runs use controlled CPU experts,
not production hybrid placement or actual MTP drafting.

The graph-recreation generation also passes all three64-case native and
all-API checks with CUDA graphs disabled. Its two ordinary21-check lifecycle
runs preserve all8,194,560 teacher logits per run, baseline logits and route
files byte-for-byte against the preceding frozen generation. The production
loader remains closed.
Fresh normal regression checks pass13/13 after the graph fix; the default
BF16 CTest skips its opt-in path, whose positive actual-weight checks are the
three native runs above. A separate synthetic-weight positive run also passes
64cases on the RTX2080SUPER.

The model sampler diagnostic supports `ESE_SAMPLER_TIMING=1` to record
synchronized scalar, checkpoint, verification-decode, sampling and commit
phase costs. Drafts are forced from the scalar oracle, which also changes cache
state before verification. These costs exclude draft generation and validation
copies; they do not establish speculative accepted-token throughput. All280
phase records are retained. Rejection-commit medians are696.68..813.63ms across
the four runs, compared with0.018..0.023ms for full acceptance. These controlled
CPU-expert/forced-draft observations identify base replay as an investigation
target, not an overall speed comparison. Qualification overlapped other local
builds/tests; they are not clean benchmark samples.

Qualified GGML/source snapshots for this stage are frozen in the evidence
directory. Subsequent PLE-prefix and allocation-binding work is qualified
separately; it does not retroactively extend this stage's coverage.
Broader shapes/formats/devices, upstream full-model parity,
production hybrid arithmetic, peak resource accounting, long-context/serving
stress and accepted-token performance remain open. The production architecture
gate and launchers remain unchanged.
