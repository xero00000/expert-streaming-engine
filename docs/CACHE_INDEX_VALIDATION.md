# Persistent cache indexes: validation checkpoint

September 4–5, 2026. Artifacts: `/home/xero/Desktop/test/results/benchmarks/ese-cache-index-20260904`.

After the first allocator improvement passed its 50-run comparison, main-thread CPU sampling still found roughly 30% of decode samples in named allocation-planner routines. A shared interval-sorting routine accounted for another 14–16%; without call stacks, that generic routine cannot be attributed entirely to the planner. These are CPU sample fractions, not fractions of wall-clock latency. Lower-rate profiling reproduced the finding, with much smaller plain/profile timing differences. See `../ese-host-profile-20260904/low-rate.json` relative to these artifacts.

The candidate maintains indexes by arena offset and by deterministic recency order. Allocation planning merges the small pending-reservation set with already ordered resident intervals and visits eligible entries in their existing eviction order. Hits update recency using extracted tree nodes, avoiding new node allocations. Publication prepares index nodes before retiring victims; erase and clear remove the corresponding index entries. Existing reservation, pin, readiness, and rollback rules remain in force.

## Completed correctness evidence

- 10,000 metadata snapshots match baseline placement, victim order, and pending-reservation decisions exactly, including fragmentation, pins, overlapping reservations, and own-reservation replanning.
- 30,000 public cache-operation steps match baseline outcomes and counters exactly. The test checks held payload bytes, source-read failures, eviction, erase, clear, and index consistency after every step.
- Allocation-failure injection walks 32 allocation positions in cold admissions. Failed admissions retain old payloads, release reservations, and leave indexes consistent. Successful admissions preserve held leases.
- The API fixture passes strict ASAN/UBSAN with exact traces and empty sanitizer stderr.
- Existing CPU, strict ASAN/UBSAN, and CUDA cache suites pass, including multi-GPU publication rollback. Isolated tests link the candidate cache implementation directly against GGML, without an old `libllama` dependency.
- Integrated builds repeat the CPU, strict sanitizer, and CUDA suites. Qwen64 and GPT32 teacher-forced logits match the frozen baseline exactly, with all values finite.

The private `ram_cache` object layout changes. Rebuild code that instantiates it; do not preload the old cache library into a test binary compiled against the new private header. Model comparison probes use the public llama API and do not import cache-class symbols. Validation-only friend declarations remain outside the engine.

## Performance gate

Planner-only tests improve free admissions about 3x and full-cache single-victim planning about 7–13x. Those timings exclude the cost of maintaining indexes during hits and publication, and cannot establish model speed. Two additional tree nodes per resident entry also add metadata overhead, which must be considered alongside payload/staging bounds and measured process memory.

The five-prompt paired 256-token NVMe screen completed all ten runs. Median paired throughput ratio is 1.3568, ready-TTFT ratio is 0.8633, and P95/P99 gap ratios are 0.7988/0.7942. Every prompt improves; paired output tokens and all measured allocation/transfer counters match. Both versions performed substantial physical NVMe reads. See `screen-ab-summary.json`. The matched comparison is now extending these pairs to five repeats per prompt, reversing order in repeats 1/3 and balancing the final repeat. The sustained result is pending. Frozen baseline libraries include payload retention, the first allocator improvement, and the FP16 callback fix. Candidate and baseline GGML libraries are byte-identical; the change is in the host cache implementation. Commands retain the same model, quantization, GPU selection, context, and RAM/VRAM budgets. Source warming is attempted before each process, but actual physical reads are reported rather than assuming the source stays warm. Broader timing validation and filled-context/soak tests remain open.
