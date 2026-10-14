# Accepted-prefix journals across verification chunks

The experimental Qwen journal now appends complete state and logits from each
microbatch into its bounded eight-input window. It also accepts successive
explicit contiguous decode calls at the same checkpoint base. Every call must
request all logits and fit the remaining capacity. The CUDA/host graph reuse
key includes the current journal-column offset; reused graphs cannot overwrite
an earlier chunk. Restoration synchronizes the backends and reads the retained
logit bank, rather than the latest decode call's output buffer.

Evidence: `/home/xero/Desktop/test/results/ese-qwen-prefix-chunks-20261011`.
All114native cases pass: microbatch1/2/4at prefix8, microbatch4at prefix32,
separate two-input calls with microbatch2, and separate three-input calls with
microbatch1. Each run checks19accepted cuts across depths1/2/4/8, including
zero/full acceptance, poisoned EOS suffixes, invalid restore requests and exact
complete sequence, continuation logits/MoE routes and retained outputs without
replay. Caller-count assertions verify the separate calls. The15kernel/sequence
regressions and5ASan/UBSan/leak checks pass. These are controlled CPU-expert model
comparisons; full upstream/production-hybrid correctness remains separate.

The full-model unsuppressed all-API memory run uses one isolated RTX3080 and
compares with its own scalar teacher. It remains in progress. A later-chunk
abort probe separately arms the public CPU abort callback after one complete
microbatch, rejects uncompleted rows and restores each completed prefix. Its
qualification is pending. Ordinary teacher controls for this generation still
need comparison; the preceding single-batch generation's controls are finalized
separately in `ese-qwen-direct-prefix-20261011`.

Eight-input reservation is1021.66MiB for this artifact, including7.58MiB of
additional verification logits. The2GiB cap covers journal/base buffers and
logits; transient graph scratch, metadata and other caches remain separate
resource scopes. The path remains opt-in and single-sequence. Sequence banks,
MTP hidden rows, broader memory/failure/serving coverage and accepted-token
throughput remain required before release.
