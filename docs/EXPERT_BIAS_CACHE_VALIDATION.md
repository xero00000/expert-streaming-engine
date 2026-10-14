# Expert bias ownership with compact GPU weights

September 4, 2026. Uncommitted checkpoint on local `ee86cae61` plus the existing development changes. Evidence and a patch relative to the preceding checkpoint are in `../../results/benchmarks/ese-expert-bias-20260904/`.

## Failure and resulting behavior

The Release GPT-OSS pilot on three GPUs, using a verified NVMe GGUF, aborted when sidecar-only staging encountered `blk.0.ffn_up_exps.bias`. The scheduler treated every host weight input of a routed split as a streamable expert matrix. Its resident two-dimensional bias table could not satisfy the matrix lease path.

Simply exempting biases from that assertion would not be correct: fused up/gate computation shares one ID tensor for weights and biases. Compact weight slots are different from the original expert IDs in the resident bias table.

The copy path now restricts expert streaming to recognized three-dimensional expert matrices. Bias tables follow ordinary resident-tensor copies and remain part of model/scheduler memory accounting. For decode with a published compact-cache policy and host expert weights, split up/gate graph construction uses separate bias gathers with the original route IDs. Weight operations alone receive remapped slot IDs. The sidecar requirement for expert weight payloads remains enforced.

Biased fused operations reaching compact staging fail explicitly. Combined up/gate matrices with fused biases remain unsupported in this mode; they are not silently assigned incorrect bias IDs. Restoring fused execution requires independent bias IDs or correctly budgeted compact bias storage.

## Validation

- Release GPU cache tests pass separately on RTX 3060 Ti, RTX 2080 SUPER and RTX 3080, despite the NVML mismatch. Device UUIDs and exit codes are recorded in `gpu-tests.json`.
- The new bias regression has eight logical experts, four physical slots, five routes, reversed load completions and six evictions. Its outputs equal both the uncached result and an analytical oracle exactly. Distinct expert-specific weights and biases detect accidental use of slot IDs for logical gathers.
- Exact diagonal arithmetic deliberately isolates ownership from accumulation precision. An initial dense F16 diagnostic showed resident/cache differences at roughly half-precision scale; that observation is retained in `parity-diagnostic.log` and is not treated as a general numerical clearance.
- CPU and ASAN/UBSAN cache tests pass. Release and Debug test logs are retained alongside build logs. No CUDA memory-sanitizer result is claimed.
- The corrected GPT-OSS pilot completes eight generated tokens, with 2,070 lease-backed uploads, 450 evictions, zero forced original-weight fallbacks, and zero outstanding loads or leases at teardown. Its RAM and staging peaks stay within 2 GiB and 64 MiB; GPU caches are 1 GiB per device.

The pilot uses only a five-token prompt and a deliberately constrained cache. It is a correctness and pressure check, not a sustained speed result. Its approximately 0.625 tok/s and 12.6 GB of physical decode reads describe that specific NVMe run, whose OS-cache state was not reset, not the engine's tuned performance. This configuration must not be compared numerically with the historical CPU/deferred-plus-resident-tail benchmark.

Model identity and NVMe location are recorded in `../ese-release-workloads-20260904/nvme-models.json`. Rotational-drive originals are archival; no engine benchmark in this checkpoint runs from them. Moving a model onto its intended execution storage is not counted as an engine optimization.

## Remaining gates

The teacher-forced GPT-OSS comparison in `model-logit-parity.json` completes 32 steps on the same token history. Synchronous and two-worker cached GPU runs have bit-for-bit identical logits, all finite. CPU expert execution with fusion disabled has the same argmax at all 32 steps, but differs from GPU logits (maximum absolute difference 1.52734375; RMS 0.07758734). This is not exact cross-placement parity or a general quality clearance. The logit probe has no intermediate capture callback, and these diagnostic runs are not timing evidence. Performance acceptance still requires sustained, paired Release runs with fixed model, actual context fill, GPU subset, cache budgets, prompts and generated-token count. Separate startup/TTFT/decode and report actual storage reads, H2D bytes, payload-ready hits and waiting time. Queued loads remain opt-in; this checkpoint does not promote a faster default.
