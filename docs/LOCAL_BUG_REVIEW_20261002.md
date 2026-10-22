# Local bug review — 2026-10-02

Scope: latest desktop worktree (ESE-current / ESE-v0.2.1-qwen4exp-plan), focused on changed GGUF loading, expert file reads/cache, resource planning, and Python launcher. This is not a complete audit of all unpublished CUDA, experimental architecture, Studio, or Android code. No commit or push was performed.

## Fixed

### GGUF unsupported in-range tensor types crash the parser

The exact Strata Coder IQ1_M GGUF contains nine type-42 tensors. ESE's sparse type table has no implementation for that ID, but the parser performed modulo by its zero block size. A real-model probe crashed with SIGFPE; GDB located gguf_init_from_file. The parser now rejects absent block/type sizes before arithmetic. The real-model probe now exits cleanly with an unsupported-type diagnostic.

`test-gguf-unsupported-type` validates supported F32 and unimplemented IDs 4, 5, 42. Qwen4Exp's architecture rejection remains intact; this is not model support.

### io_uring completion errors poison subsequent reads

Both single and batch reads threw on failed/short CQEs before consuming them. A regression that truncates a temporary source after construction produced a short read and then failed an unrelated valid read against the stale CQE. Single reads now copy and consume the CQE before validation; batches copy and retire the completed group before validating each result. Single reads also validate completion identity.

The regression covers single-read failure/recovery followed by batch failure/recovery. Truncation is fault injection, not a supported model mutation workflow.

### Test harness concealed storage failures / failed without a GPU

`test_file_storage_backends` caught runtime errors throughout the io_uring test and labeled them backend-unavailable. Only io_uring_setup unavailability is now skippable; operational errors fail the test.

The CUDA frequency-retention comparison ran even with zero visible devices, compared two empty runs, and asserted that zero uploads must be less than zero. It now follows the other CUDA tests' no-device guard.

### Invalid GGUF alignment could crash or misalign parsing

The parser trusted general.alignment before modulo and GGML_PAD arithmetic. Added rejection of zero, non-power-of-two alignment, and non-UINT32 metadata. Fixture coverage tests valid 32, zero, 3, and string-valued alignment. This is parser hardening based on the identified arithmetic path, not a failure observed in the user's model.

## Remaining source-level concern

`src/llama-expert-cache.cpp`, file_source::impl::read_many: short io_uring submission or submit/wait syscall errors still throw without an explicit drain/cancel protocol. If requests have already been accepted, caller-owned destinations can be released or reused while I/O may still reference them; the shared ring can also retain pending submissions/completions. The corrected CQE-result handling does not address these earlier paths. This needs syscall fault injection plus a carefully validated submission/drain or fail-closed ring lifecycle. It was not reproduced or repaired during this pass and must not be described as tested-safe.

Other deferred areas: malformed tensor metadata still uses assertions; full-model CUDA numerical parity, multi-GPU error paths, experimental DeepSeek/Qwen support, transaction stress, sanitizer runs, and Studio/Android code were not comprehensively reviewed here.

## Validation

- Existing Release CUDA 75/86 configuration rebuilt llama-server and focused tests successfully.
- Four focused CTests passed with GPUs hidden: expert-cache, resource-planner, GGUF regression, Qwen4Exp rejection.
- Expert-cache suite passed on the available RTX 2080 SUPER via CUDA_VISIBLE_DEVICES=1; Strata's Ampere pair was left running.
- Native io_uring failure/recovery regression executed successfully on this host (not skipped).
- Python unittest discovery: 85 tests passed, including 29 launcher tests.
- git diff --check passed.

No full sanitizer, full CTest, multi-GPU, long-context inference, or task-quality validation was performed. The Release compiler emits existing enum-switch warnings in ggml_compute_forward_clamp; these were not changed.
