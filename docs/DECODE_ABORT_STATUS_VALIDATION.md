# Distinct graph-abort decode status — 2026-10-12

An aborted graph now returns `LLAMA_DECODE_ERROR_ABORTED` (-4) from
`llama_decode`. Previously it propagated `GGML_STATUS_ABORTED` (1), aliasing the
public warning for KV-slot exhaustion. A caller could therefore reduce the
batch and retry a partially executed graph. KV-slot exhaustion still returns
1; the existing -3 completed-chunk cancellation result is unchanged.

Evidence: `/home/xero/Desktop/test/results/ese-decode-abort-status-20261012`.

The new public-API test loads a deterministic nonzero tiny Llama GGUF. It invokes
real CPU graph abort callbacks at early and later poll thresholds, checks the
distinct negative status, explicitly restores a checkpoint and reproduces the
saved session and next logits exactly. A separate context fills its KV capacity
and confirms that actual exhaustion still returns positive 1.

The test passes in Release and CPU ASAN/UBSAN/leak builds. The same executable
against the frozen pre-fix runtime fails at the abort-status assertion, providing
a negative control without changing the public parameter layouts. The broader
current regression groups pass 28 Release and 20 CPU sanitizer tests. An initial
Release run hid every GPU and failed the PLE test's required-device check; the
retained rerun on the Turing GPU passes all 28. This was a test configuration
failure, not a PLE numerical failure.

The status change does not automatically roll back an aborted graph. As before,
state may have been written; callers must restore a checkpoint or clear the
affected context before replay. The CPU test qualifies explicit checkpoint
recovery. A separate actual-model control on the selected 3060 Ti/3080 pair also
passes all ten accepted-prefix cases across microbatches 1, 2 and 4. It interrupts
the second chunk after the first completes, requires status -4, rejects the
incomplete checkpoint row, then explicitly restores a completed prefix.
Serialized sequence state, last logits and next-token logits/routes match a
separately decoded teacher exactly. This uses F32 KV, Flash Attention off, CPU
experts, a 64 MiB expert cache and 4 MiB staging. It establishes recovery within
the same diagnostic runtime; it does not establish upstream full-model parity.
The microbatch-2 control also passes all three recovery cases under an all-API,
full-leak CUDA memory check: zero errors and zero leaked bytes, terminal exit 0.

HTTP cancellation, concurrent-session behavior and driver allocation failure
remain unqualified. Ordinary successful graph arithmetic is unchanged.
The normal Qwen architecture gate remains closed.

The server's separate [decode recovery validation](SERVER_DECODE_RECOVERY_VALIDATION.md)
now covers automatic context invalidation after actual CPU graph aborts,
system-prefix reconstruction and correct error classification. It clears the
shared context rather than providing per-request checkpoint rollback.
