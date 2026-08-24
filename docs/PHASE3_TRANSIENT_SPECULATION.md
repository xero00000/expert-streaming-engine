# Phase 3 transient modules and adaptive speculation

## Scope

Phase 3 shares an explicitly bounded device-memory allowance among reconstructible
modules such as MTP/draft heads, vision and audio encoders, rerankers, embedding
heads, and temporary LoRAs. It also makes speculative depth depend on measured net
throughput and enables the existing ordered-embedding metadata as a mapped draft
vocabulary.

## Failure-atomic transient transactions

`common_transient_module_manager` owns residency state and per-device budgets.
Request and policy changes use prepared exact-owner transactions:

1. validate every requested module and reserve capacity on every affected device;
2. retain the configured device safety margin;
3. select least-recently-used victims without changing backend state;
4. prepare complete off-side owners while the live owners remain usable;
5. quiesce only affected modules and publish by no-allocation owner swaps;
6. pin every overlapping consumer into one residency epoch;
7. roll the epoch back only when every consumer fails or requests restoration,
   otherwise finalize it and retire the old owners; and
8. resume affected streams in reverse order.

Reservation or preparation failure leaves live owners untouched. Partial
publication reverses earlier swaps before returning. Once a published owner set
is visible, rollback and finalization are invariants: a callback failure is
fail-stop rather than exposing a mixed or unowned module set. Value-only policy
metadata and exact MTP/mmproj owners survive scratch-graph and request lifetimes.

Structured telemetry reports transactions, swaps, rollbacks, failures, OOM
rejections, bytes moved, cumulative swap latency, current resident modules, and
resident bytes per device.

### Server integration

`llama-server` can apply the manager to a combined MTP and multimodal deployment.
The mode is opt-in and requires all four measured values:

```text
--transient-vram-budget-mib N
--transient-vram-reserve-mib N
--transient-mtp-mib N
--transient-mmproj-mib N
```

The multimodal estimate must include projector weights and its peak image/audio
compute graph, not merely the projector file size. The MTP estimate must include
the companion context, sampler, KV, and compute buffers. At startup the server
creates and validates MTP, suspends it, and only then loads the projector, avoiding
simultaneous transient residency. A request lease is acquired before slot launch
and released with that slot on success, cancellation, or failure. Multimodal prompt
parsing holds an overlapping projector lease so another HTTP worker cannot evict
the context while it creates media chunks.

Pre-tokenization owner changes are routed through the inference owner thread.
Each HTTP batch transfers one lease group into its queued tasks: no launched task
restores the exact prior owner, while any launched task commits the new owner so a
stale MTP cache cannot be paired with a changed target context. Overlapping image
requests share that decision through a residency epoch. Temporary contention is
deferred until a slot or residency change wakes it. Parse, launch, cancellation,
and graceful-shutdown paths resolve every pin before backend teardown.

Runtime policy is explicit: `off`, `shared`, `mtp-only`, or
`multimodal-only`. The resource controller prepares a complete policy candidate,
publishes it with KV/expert transactions when requested, and exposes the policy,
configured bounds, residency, pins, transactions, swaps, rollback count, and
latency through `/v1/ese/resources`. `/props` publishes modality capability with
the same mutex-protected plan snapshot.

Plain server operation is unchanged when the budget is zero. Admission rejects a
module estimate that exceeds `budget - reserve`; backend allocation failure is
reported separately, because the fixed target model and its execution graphs are
outside this transient allowance.

## Adaptive MTP depth

MTP autotuning evaluates depths `0..n_max`; depth zero is a real target-only arm.
The reward is complete generation-step tokens per second, so it includes draft
compute, target verification, and accepted-token benefit. After a minimum sample
count, depths below 90 percent of measured target-only throughput are quarantined.
Periodic recovery probes allow a depth to return after the workload changes.

The speculative metrics snapshot and `llama-spec-bench` JSON expose current and
best depth, net tokens/s versus target-only, target-only and speculative selections,
quarantines, probes, and the latest automatic retune/disable reason. Acceptance,
draft time, verification/accept time, and accepted/drafted counts by depth remain
separate fields.

## Mapped draft vocabulary provenance

The mapped head is enabled only when all of the following GGUF/model facts agree:

- `*.use_ordered_embeddings = true`;
- `*.centroid_count` and `*.centroid_top_k` are positive and shape-compatible;
- `mtp_centroids.weight` has `[hidden, centroid_count]` logical dimensions; and
- `mtp_token_ordering.weight` is an I32 permutation containing one explicit
  draft-to-target token id for every target-vocabulary entry.

The ordering comes from the model publisher's frequency/centroid analysis; ESE
does not silently synthesize or reorder it. The graph selects the best centroids,
gathers their target token ids and output rows, computes only those candidate
logits, then scatters them into a full target-vocabulary row. Non-candidates are
masked only in the draft row. The target model always verifies with its untouched
full vocabulary, so unsupported draft tokens remain available and temperature-zero
target output is byte-identical to non-speculative decoding.

Frequency-ranked maps are workload-biased. Natural-language frequency can improve
prose drafting while reducing coverage for code symbols, tool syntax, less common
languages, or domain terms. Validation therefore reports code, prose, tool,
multilingual, and long-context panels independently; one aggregate acceptance
number is not sufficient.

## Validation matrix

Before Phase 3 is mergeable, record the following separately:

- one-slot and multi-slot transaction tests;
- capacity rejection, activation failure, request failure, and complete restore;
- image request followed by text/MTP generation with no stale state;
- failed parse, failed launch, deferred cancellation, overlapping batches, and
  shutdown with zero remaining pins or leases;
- all seven non-empty KV/expert/transient rebalance scopes and every reversible
  publication fault boundary;
- mapped-head CPU and CUDA operator checks;
- temperature-zero output bytes with mapped vocabulary on and off;
- code, prose, tool, multilingual, and long-context acceptance panels; and
- peak memory delta and net tokens/s delta versus no speculation.

CPU unit tests cover transaction atomicity, multi-device capacity, target-only MTP
selection, profitable depth selection, mapped-id scatter, and greedy full-target
verification. End-to-end model/hardware evidence is recorded with the consolidated
Phase 3 pull request.
