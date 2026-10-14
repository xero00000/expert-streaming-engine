# Original Qwen initialization under memory pressure — 2026-10-12

The original read-only, two-shard Qwen artifact initializes on the selected
3060 Ti/3080 pair with a requested microbatch of 1024 and an empty context
capacity of 131072 tokens. This check uses host Q8 KV, the current F32 indexer,
CPU expert execution and the diagnostic loader gate. It executes no tokens.

| Run | GPU reserve | Effective microbatch | Budget rejections | Exit |
| --- | ---: | ---: | ---: | ---: |
| Native baseline | 700 MiB | 1024 | 0 | 0 |
| Native pressure | 4500 MiB | 256 | 2 | 0 |
| All-API/leak sanitizer pressure | 4500 MiB | 128 | 3 | 0 |

The native pressure run rejects 1024 and 512 before admitting 256. The
instrumented run also rejects 256 before admitting 128. The latter reports
zero CUDA API/memory errors and zero leaked allocations. No allocation failure
is reported by the graph arena counters in these runs.

At 50 ms target sampling intervals, native baseline process VRAM peaks are
5,676,990,464 and 5,974,786,048 bytes on the selected GPUs. Native pressure peaks
are 3,370,123,264 and 3,789,553,664 bytes. These are sampled process peaks;
transient allocations between samples can be missed. Whole-device used memory
includes other processes and driver/display allocations. Native sampled RSS
peaks are 30,341,120,000 and 29,596,270,592 bytes respectively. Sanitizer RSS
records only its parent process, so it does not measure the model's host RSS.

The probe holds initialized buffers for 250 ms to permit external sampling,
then frees the context, model and backend. Its elapsed time includes setup,
this hold and teardown; it is not a throughput or time-to-first-token result.
Shard size/inode/device/modification-time identities match before and after
each run. This stage does not repeat full payload hashing.

Evidence: `/home/xero/Desktop/test/results/ese-qwen-prefill-resources-20261012`.
The bundle freezes the diagnostic runtime, source digest, owned probe patch,
argv/environment, logs, samples, terminal receipts and qualification scope.

This establishes initialization/backoff on the original artifact under current
settings. Current weight placement and F32 indexer differ from the historical
failed profile; the result does not explain or reproduce that historical
performance gap. It does not qualify actual prefill, filled 64K/128K contexts,
aggregate GPU budgets, output parity, throughput or the complete release.
The original 33-token numerical check remains failed and normal loading gated.
