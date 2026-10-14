# Server decode failure recovery — 2026-10-12

The server now treats graph aborts and other fatal decode errors as failures of
the shared decode context. It synchronizes and clears KV, cached token prefixes,
saved prompt state, checkpoints, draft indexes and rewind buffers for slots
using that context. A shared system prefix is rebuilt before request decoding
resumes. Idle slots receive no error response; slots belonging to an independent
context retain their state.

This clears damaged state rather than restoring every request to its previous
checkpoint. Clearing only the sequence in the failed batch was insufficient in
the reproduced case: another idle slot's prefix in the same context also changed.

Only positive result 1, KV-slot exhaustion, can trigger a smaller-batch retry.
Result -4 reports a graph abort; -2 reports working-memory allocation failure;
other failures receive a decode error. A failure while rebuilding the system
prompt terminates affected active requests instead of continuing with an
incomplete prefix.

Queued responses also have a virtual base destructor. The sanitizer regression
found that deleting an error response through the previous base pointer caused
a `new-delete-type-mismatch`. The corrected queue test verifies destruction of
resources owned by a derived error response.

Evidence directory:
`/home/xero/Desktop/test/results/ese-server-decode-failure-20261012`.

## Verified checks

`test-server-decode-recovery` generates a nonzero three-layer CPU Llama model.
It runs real graph abort callbacks at poll thresholds 1 and 20, with and without
a shared system prefix, plus an abort during system-prefix reconstruction.
All five cases require cleared cache/draft/rewind metadata and KV, one error for
the failed active request, unchanged serialized state in an independent context,
and byte-exact logits after recomputation with the original batch shape.

The Release and CPU AddressSanitizer/UndefinedBehaviorSanitizer runs each pass
three tests: server decode recovery, server token/response handling, and public
decode-status/capacity distinction. Leak checking and halt-on-error are enabled
for the sanitizer run. The initial sanitizer ownership failure is retained.

Six HTTP controls pass on the final frozen server:

- Four real CPU graph aborts: two repetitions without a system prefix and two
  with one. Each returns HTTP 500 with the graph-abort message. The next request
  fully recomputes its 12-token prompt and matches its clean teacher exactly.
  The other slot's eight-token prompt is invalidated and recomputed exactly.
- One simulated -2 result verifies allocation-error classification and recovery.
- One simulated positive 2 verifies that a non-capacity result is fatal and is
  not retried by reducing the batch.

The simulated results test server control flow; they do not establish recovery
from an actual allocator or driver failure. The actual abort shim calls the
normal decoder with a CPU graph abort callback and records result -4 at poll 20.
Servers bind only to loopback and are terminated by their owning harness.

The retained old-server HTTP control returns the misleading prompt/KV-capacity
error after the same real graph abort. Its next request reuses the failed prefix
and differs from clean recomputation. Two initial fixture/configuration failures
(CPU F32 KV with Flash Attention enabled, and missing standalone tokenizer
pieces for the system prefix) are retained separately from runtime failures.

## Copy-layout control and scope

An additional simulated capacity-1 control successfully retries with a smaller
batch but fails the output comparison. A direct core-decoder probe reproduces
the discrepancy without the server, both with and without an idle shared
sequence, and with and without requesting an intermediate output. On this
synthetic F32 model, an 8+4 split differs from a single 12-token decode by up to
0.187703 in the final logits. That original failed comparison is retained.
The separate [CPU transpose-copy validation](CPU_TRANSPOSE_COPY_VALIDATION.md)
identifies and repairs a strided destination copy. With that repair, the
full/split model logits are byte-exact and all seven HTTP controls pass,
including the capacity retry and an unchanged idle-slot saved prefix.

The normal core libraries are byte-identical across the initial server-only
qualification. The subsequent CPU copy guard is recorded separately; no
projection arithmetic or reference tolerance is changed. This evidence
does not qualify Qwen HTTP serving, real MTP recovery, simultaneous HTTP clients,
disconnect cancellation, driver failure, or end-to-end throughput. The original
Qwen 33-token numerical gate remains closed.

To run the generated-model regressions:

```sh
cmake --build build_cuda_release --target test-server-decode-recovery test-server-tokens test-decode-status --parallel 4
ctest --test-dir build_cuda_release --output-on-failure -R '^(test-server-decode-recovery|test-server-tokens|test-decode-status)$'
```
