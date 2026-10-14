# Persistent Qwen decode resource check — 2026-10-12

One loaded diagnostic context completes 1805.87 seconds of repeated nine-token
decoding and resets: 189 rounds, 1701 decoded tokens and 1692 exact replay
comparisons. Every replay matches its baseline logits and all captured routes
bit-for-bit. The baseline passes 2234880 logits at the existing
`2e-5 * (1 + abs(reference))` bound and all 432 ordered expert selections against
the pinned nine-token reference. Normal Qwen loading remains gated.

Evidence: `/home/xero/Desktop/test/results/ese-qwen-soak-20261012`,
`qualification-final.json`. Both original model shards remain read-only;
size/inode/device/mtime identities match before and after the run.

## Workload and memory

The selected 3060 Ti/3080 pair runs dense operations with F32 KV, Flash Attention
off, context capacity 4096 and CPU experts. Expert RAM is capped at 64 MiB with
4 MiB staging. The probe replays a fixed nine-token input in one context, clears
it between rounds and validates output ownership, positions and resource
snapshots. It does not reload the model between rounds.

Every tracked native inventory entry remains unchanged after round two. CUDA
workspace backing stays 125829120 bytes per backend; active pool leases are
zero at each checkpoint. Graph/pool shared reservations remain within their
current caps. Expert resident bytes remain within their cap, active expert
leases drain after each round, and final teardown reports no active loads,
leases, staging use or pending reservations.

At a 50 ms target sample interval, native process RSS peaks at 26724675584 bytes.
Owned process VRAM peaks at 2596274176 and 2889875456 bytes. These are sampled
process measurements, separate from other processes' device usage; they cannot
prove every shorter allocation peak. Tracked mapped model spans are virtual
address ranges, not resident RAM. Background CPU compilation occurred during
part of the run; this is resource/replay evidence, not a performance baseline.

The expert cache records zero retained hits, 2449440 misses and
1670844672000 logical pread bytes. That byte counter is not physical SSD
traffic. Its read timer totals 184.55 seconds, about 10.22% of the instrumented
decode-and-capture time. This ratio does not establish critical-path attribution
or an achievable speedup; it includes this diagnostic's callbacks and workload.

## Harness checks and limits

A short control completes three rounds/27 tokens, including 18 exact replays.
Invalid durations and incompatible trace/prefill modes reject before loading.
The initial header-collision build and two incorrect empty-cache assertions are
retained as harness failures. Both failed runs stopped before any decode. The
final reset assertion checks used cells and cleared output ownership directly.

The long test uses frozen libraries from before the separate graph-abort-status
change. It never injects cancellation or allocation failure. That later change
has its own [decode-status checks](DECODE_ABORT_STATUS_VALIDATION.md).

This qualifies sustained repeated short-prefix decoding on one context and the
selected Ampere pair. It does not qualify growing/filling a 64K context,
autoregressive sampling, concurrent sessions, server disconnect handling,
general failure-atomic rollback, Turing whole-model execution or a complete
aggregate allocation budget. The original 33-token numerical gate remains
failed; no throughput claim or production promotion follows from this test.
