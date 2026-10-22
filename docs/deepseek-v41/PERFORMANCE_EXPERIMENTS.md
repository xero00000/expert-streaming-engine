# V4.1 scheduling experiments — September 12, 2026

The current bottleneck is expert delivery. The first instrumented decode run
averaged 555 ms of host scheduling per step, including 353 ms waiting for
expert leases and 79 ms submitting uploads. It read 475 MB physically from
storage and uploaded 1.013 GB to the GPU per decode step. These are measured
process counters and existing host waits, not estimates from GPU utilization.

The machine has three GPUs: 8 GiB RTX 3060 Ti, 8 GiB RTX 2080 SUPER and 10 GiB
RTX 3080. The fast layout places Q8 dense projections and adjacent pointwise
operations on CUDA0, the vocabulary head on CUDA1, and sensitive nonquantized
projections plus routed expert computation on CUDA2. This makes room for about
7 GiB of expert payloads on CUDA2. Moving whole layers changed numerical
results; the selected placement preserved the verified Q8 computation.

## Results and measurement limits

Every model timing run was serialized with other model timing runs and builds.
The OS page cache was not reset. Desktop applications continued running, and
the realized GPU cache sometimes fell from 754 to 751 slots as free VRAM
changed. These are local workload measurements, not a hardware maximum or a
guaranteed long-context rate. Reported generation rates use the CLI token
counter; the trace counts actual forward calls separately.

| Experiment | Observed result | Decision |
| --- | --- | --- |
| Grouped dense placement, eight loader workers, 8 GiB RAM cache | 1.81–1.87 tokens/s on the sky prompt | Useful baseline; preserves numerical results |
| Reduce explicit RAM cache to 512 MiB | Initial repeats: 2.05 and 2.10 tokens/s; host RSS about 10 GiB instead of 17.5 GiB | Leaves more memory for the OS file cache |
| Pin only the 512 MiB RAM arena | 1.93 and 2.03 tokens/s; later paired pageable controls were 1.61 and 1.60 | Bounded asynchronous upload option; see the longer comparison in the validation JSON |
| Predict the previous token's expert set | 38.1% total recall, 6.8% recall of VRAM misses | Insufficient signal for broad prefetch |
| Predict experts used in three consecutive tokens | 73.3% precision, only 0.1% recall of VRAM misses | Mostly predicts weights already cached |
| Early router, without reads | 82.8% overall precision, 71.3% miss precision, 81.1% miss recall | Useful predictor; median read head start only 1.8 ms |
| Early router with 256 MiB advice budget | Ungated run 1.93 tokens/s; gated pinned run 1.60 | Experimental; no reliable end-to-end win |
| Random-access file advice | 1.68 tokens/s in the tested pinned run | Keep ordinary file advice as the default |
| Lossless Zstandard on 18 full expert components | About 0.9% byte reduction; no duplicate 4 KiB blocks in the sample | No evidence here for a large ordinary compression win |

Later runs varied considerably even with identical expert routes. The first
2.10 tokens/s result should therefore be read alongside its controls, not used
as a universal speed claim. Full commands, environments, physical I/O,
throughput and output hashes are recorded in `speed4-validation.json` and the
individual result directories under
`/home/xero/Desktop/test/results/deepseek-v41-20260910`.

Pinning reduced host upload submission time from roughly 80 ms to 2–3 ms per
step, but some of that time reappeared as waiting for the next storage
completion. Completion-as-ready loading already overlaps reads with uploads.
Removing a host wait does not necessarily shorten the execution path.

## The early-router experiment

Before a layer's attention runs, the prototype applies that layer's learned
hyper-connection residual mixing, omits the uncomputed attention contribution,
then applies the actual FFN normalization and router weights. The resulting
expert IDs are predictions only. The normal attention, router, top-six expert
selection, routing weights and expert arithmetic all execute normally.

Predictions exclude resident GPU components. A background worker advises exact
file ranges through `POSIX_FADV_WILLNEED`; it does not insert speculative
payloads into the explicit RAM cache or alter its retention counters. It yields
while the demand queue is occupied. Expired queued hints are cancelled, while
already issued OS advice is best effort and cannot be recalled. The software
queue holds at most 64 requests, deduplication metadata is capped at 4096 keys,
and each graph pass has a byte budget.

The optional gate learns a per-layer precision estimate from *previous*
predictions that missed VRAM and were checked against actual routing. At least
16 checked predictions are required; the estimate uses one success and one
failure as smoothing, and ages its counts. The default threshold is 0.85.
It is an empirical precision threshold, not a statistical confidence interval.
Replay of all 1,443 forecast events in the gated run verified that the estimate
used only preceding actual routes.

The remaining limitation is scheduling distance: a useful prediction delivered
only 1.8 ms early cannot hide every multi-megabyte read. A worthwhile next
experiment is an earlier forecast whose objective is *avoided miss latency*,
with admission accounting for false reads and current device bandwidth. This
implementation does not establish that the idea is unprecedented.

## Diagnostics

Add these to the normal smoke runner as needed:

```text
--expert-trace /absolute/path/trace.jsonl
--early-router probe
--early-router advice --prefetch-mib 256 --prefetch-confidence 0.85
--pinned-ram                  # RAM cache must be at most 1 GiB
--file-advice random          # experimental, Linux only
```

The early router currently runs only for single-token decode. Prefill retains
the normal graph. Trace files append; use a fresh path for each run. Summarize
a completed trace with:

```bash
python3 tools/deepseek_v41_trace.py /absolute/path/trace.jsonl \
  --output /absolute/path/trace-summary.json
```

`host_schedule_ns` ends when graph submission returns; it is not a complete
GPU timeline. Stage elapsed times contain their sub-timers. Worker storage
times overlap one another and must not be added to wall time. Physical reads
are Linux process `read_bytes` deltas, including speculative activity.

The bounded compression probe reads 18 components across six layers and checks
byte-for-byte decompression at both Zstandard levels:

```bash
python3 tools/deepseek_v41_compression_probe.py \
  --package /home/xero/Desktop/test/models/deepseek-v41-native-q8dense \
  --output /absolute/path/compression.json
```

This sample does not rule out specialized transforms, cross-expert residual
coding or redundancy elsewhere in the model.

## Correctness

All 54 positions, including the 40 answer positions, reproduced the previous
verified Q8 profile's logits bit for bit with pinned RAM, early advice and
random file advice enabled together. This verifies these execution changes;
the existing Q8 dense and mixed-Q2/affine2 model recipe remains approximate
relative to the released higher-precision weights.

The cache tests cover advice bounds, demand priority, expiry, descriptor/source
identity, nonpromotion of resident data and external-arena ownership. They pass
under AddressSanitizer and UndefinedBehaviorSanitizer. All 15 native tests also
pass with the early-router graph enabled, including rollback, batching and
target-feature capture. See `speed4-all-features-comparison.json` and the
`speed4-final-asan-*` logs in the result directory.

## What the supplied research changes

### Follow-up: spare GPU capacity and loading concurrency

The next five serialized sky-prompt runs tested the existing
`ESE_V41_EXPERT_LAYER_COUNT` placement control. With
`ESE_V41_EXPERT_DEVICE=CUDA1`, it pins routed expert matmuls for the first N
layers to the 2080 SUPER; the trace confirmed that remaining layers execute
on CUDA2. Dense, router and head placement stayed as in the documented profile.
The 2080 SUPER allocated a separate cache of 592 slots (5,632 MiB capacity).

| Run, in execution order | Decode tokens/s | Decode expert misses | Physical MB/step | Upload MB/step |
|---|---:|---:|---:|---:|
| First 4 layers' experts on CUDA1 | 1.433 | 3,374 | 561.3 | 907.6 |
| Current profile, 8 loading workers | 1.561 | 3,765 | 562.8 | 1,012.8 |
| First 8 layers' experts on CUDA1 | 1.368 | 3,186 | 545.1 | 857.1 |
| Current placement, 4 loading workers | 1.604 | 3,772 | 574.1 | 1,014.7 |
| Current profile, 8 workers repeated | 1.596 | 3,769 | 586.5 | 1,013.9 |

All five runs produced the same output and all 1,560 routed expert-ID arrays
matched. These rejected placement candidates did not receive a full-logit
comparison. MB denotes decimal megabytes; step means one of the 37 decode
graphs. CLI throughput counts 38 generation tokens, including EOS. Cache
capacity on CUDA2 varied with desktop memory use (736 slots in the eight-layer
trial). The OS cache was not reset. A 16-worker probe was rejected by the
existing 0..8 argument limit before model loading and has no timing result.

Extra VRAM reduced uploads but barely reduced physical reads in these short
runs. The four-layer trial also increased mean host scheduling from 642 to
693 ms per decode graph, despite slightly less lease waiting. This is
consistent with additional device/transfer overhead; it does not isolate
PCIe time. Neither layer split improved throughput. Four versus eight loading
workers is within observed variability. Keep the documented profile unchanged.
Exact commands, environments and process counters are in
`/home/xero/Desktop/test/results/deepseek-v41-20260910/speed5-validation.json`.

The next spare-device experiment should route **already resident, frequently
reused experts** to CUDA1 while serving cold experts on CUDA2. That requires
partitioning experts within each layer, with measured promotion costs and
causal admission decisions; the whole-layer experiments above do not implement
it. Longer runs are needed to assess whether promotion and warm-up amortize.
Other unverified candidates are routing predictions with more lead time than
the measured 1.8 ms and better accepted-token yield from speculative decoding.
Neither has a demonstrated speedup in this profile yet.

- [NeutronStar](https://github.com/scottjoyner/neutronstar/blob/main/README.md)
  reports a useful next-layer routing experiment. It motivated checking a
  learned-weight predictor, with measured miss precision and a bounded budget.
- [Pulsar](https://github.com/giannisanni/pulsar/blob/main/README.md) distinguishes
  resident placement from caching. Its spare-device expert execution is worth
  studying for the unused capacity on CUDA1. ESE's frequency cache already has
  transient admission for equal-frequency misses.
- The [llama.cpp streaming discussion](https://github.com/ggml-org/llama.cpp/discussions/23324)
  includes larger RAM caches that reduced bytes but lost speed. Our smaller
  explicit cache experiment addresses the same practical tradeoff.
- [SliceMoE](https://arxiv.org/abs/2512.12990) changes resident precision;
  its speedups are not evidence for bit-identical acceleration of this
  already low-bit expert representation.
- [OD-MoE](https://arxiv.org/abs/2512.03927) evaluates distributed expert
  prediction on different models and hardware. Its headline accuracy should
  not be assumed for this setup.

Cold CPU expert execution still pays storage-read cost when the weights are
absent from RAM. It can save the weight upload and overlap GPU work; it cannot
remove the initial read. Reduced/adaptive top-k would change the model and is
not part of the verified speed profile.
