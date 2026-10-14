# ESE v0.2.1 release notes

**Working draft — unpublished.** This document describes the current candidate.
Eight unsigned installer/package candidates have completed their recorded
packaging checks. Signing, the new filled-64K flagship demonstration and broader
release qualification remain incomplete. Experimental capabilities below are
not a declaration of production support.

## More dependable local AI on the hardware you own

ESE runs large sparse AI models by managing their data across your SSD, system
memory and graphics cards. A sparse model chooses a small group of specialists,
called experts, for each token. ESE loads and retains the experts it needs within
the available memory budget.

This update strengthens that process: independent expert reads can overlap,
cached weights survive changes between layer layouts, and failed memory
operations preserve usable state. It also improves the numerical checks behind
specialized CPU and GPU execution.

For users, these changes address three practical problems: repeatedly loading
data that should already be cached, waiting behind unrelated storage requests,
and losing a valid session when a replacement allocation or restore fails.
Their effect on response speed depends on the model, memory budget and hardware.
There is no release-wide speedup claim in this draft.

## Expert loading and cache reuse

### Independent RAM-cache reads can overlap

The RAM cache reserves space under its metadata lock, performs storage reads
outside that lock, then publishes the completed payload. A ready cache hit can
proceed while an unrelated expert is still loading. Duplicate requests for the
same immutable component share its load and completion result.

Resident and staging payload arenas remain fixed in size. A failed read releases
its reservations and preserves retained data. Expert leases protect bytes that
are still in use, including eviction candidates acquired while a read is in
flight. This cache change does not itself turn the synchronous storage reader
into a completion-driven I/O backend.

### Optional bounded expert-load workers

`--expert-load-workers N` provides a bounded queue for GPU-route cache misses.
It accepts 0–8 workers; zero retains the synchronous default. Nonzero settings
require the native RAM cache.

The queue admits work against both request slots and the existing staging-byte
limit. Completed requests retain their budget charge until consumed. Ready RAM
payloads bypass worker dispatch, and independent completions can be consumed
without waiting for the first submitted request. Lease protection continues
through the GPU upload.

The selected experts and routing weights are preserved. Current queue ownership
is serialized between GPU route batches/contexts; this is not unrestricted
concurrent-session scheduling. Worker stacks and queue metadata add host
overhead beyond the payload arena. No worker count is advertised as universally
faster, and the default remains zero.

### Calibration selects the CPU decode endpoint correctly

When calibration chooses CPU expert execution, the launcher now disables the
adaptive decode GPU cache. Previously it removed the mixed CPU/GPU split but
retained a positive GPU cache, which sent cold experts to the GPU despite the
reported CPU selection. The all-GPU endpoint retains its cache, and an explicit
mixed candidate can still be prepared for workload validation.

Eight endpoint controls cover automatic and explicit cache budgets under cache
and stream policies; the old launcher fails the CPU case. All 31 launcher tests
pass, including explicit mixed-candidate handling. This corrects command
selection; it does not establish a universal placement speedup.

### Cached payloads survive layer-layout changes

GPU cache entries now retain their actual payload layout and use fixed physical
slot strides. Selecting another layer's metadata no longer invalidates unrelated
cached weights. Uploads use the component's logical byte count at its physical
slot offset.

Cache-owned padding handling also prevents tensor alias initialization from
clearing the tail of a retained expert. Cache replacement and rollback preserve
the stored layout and payload ownership. Telemetry distinguishes usable payload
hits from metadata hits, and reports layout reloads and padding upload bytes.

Specific before/after model checks retained exact teacher-forced logits for
Qwen3.6 35B-A3B and GPT-OSS 120B. These short checks do not establish arbitrary
model, long-context or throughput equivalence.

## Memory accounting and recovery

### Clearer native allocation inventory

An additive native inventory query reports tracked memory pools by category and
scheduler backend. It covers model buffers or mapped spans, attention and
recurrent caches, checkpoint banks, graph arenas, output buffers, prepared cache
replacements, expert caches, prefill staging, handoff arenas and selected host
metadata.

CUDA workspace pools also report their physical backing, including cached
allocations retained for reuse. A private diagnostic query separates active,
cached and virtual-reservation bytes. Virtual address reservations do not count
as physical VRAM. The inventory field reports physical bytes once; these pool
statistics do not enforce an aggregate allocation limit.

An explicit private cap can reject workspace-pool growth before allocating its
physical backing. Graph execution reports allocation failure, releases temporary
leases and discards an unfinished CUDA capture so a caller can retry. VMM and
legacy component checks cover rejection/retry with capture enabled and disabled
on all three GPUs. Earlier graph nodes can still have changed session state;
this is not complete session rollback or an automatic combined memory budget.

Context planning also joins graph arenas and CUDA workspace pools under one
shared admission ledger per selected GPU. Concurrent owners reserve bytes before
allocating, so they cannot each spend the same remaining headroom. Replacement
arenas include the old allocation until it is safely retired, and cached pool
backing stays charged. This combines those compute allocations; other context,
cache, handoff and vendor allocations still prevent a complete aggregate limit.

Prefill initialization also prepares known explicit-F32 CUDA matrix workspaces
before reserving graph arenas. It refreshes headroom after pool backing and
cuBLAS handle creation, then backs off the causal microbatch when necessary.
The retained original-model pressure case now automatically selects chunk 128
from a request of 1024 and completes its 128-token input. Its final logits and
all ordered expert routes are byte-identical to the saved explicit-128 control.
The same pressure workload passes an Ampere-pair all-API/leak memory check with
zero errors/leaks and byte-identical native/instrumented outputs and routes.
The estimator covers specific matrix dispatch paths, not every CUDA workspace.

A persistent-context resource check also completes 1805.87 seconds of repeated
nine-token decode/reset cycles: 1701 tokens and 1692 exact replay comparisons.
Every tracked native inventory entry remains fixed after warmup, and expert
ownership drains. This is a bounded short-prefix diagnostic on the Ampere pair;
it does not establish filled long-context, concurrent-session or throughput
qualification.

Graph aborts now return the distinct negative `LLAMA_DECODE_ERROR_ABORTED` (-4).
Previously they returned 1, which also meant recoverable KV-slot exhaustion and
could invite a batch-size retry after partial execution. KV exhaustion still
returns 1. Callers must restore a checkpoint or clear after an aborted graph;
the new status does not provide automatic rollback. Real CPU abort/capacity
tests and an old-runtime negative control qualify this distinction. Actual Qwen
abort checks also pass all ten accepted-prefix cases across microbatches 1, 2
and 4: explicit recovery reproduces sequence state, last logits and continuation
logits/routes exactly on the selected Ampere pair. The microbatch-2 all-API/leak
CUDA check also completes with zero errors and zero leaked bytes.

The server now clears the failed shared context before reusing a slot after a
fatal decode error. This invalidates cached prefixes, checkpoints, draft state
and rewind buffers, then rebuilds any shared system prompt. Idle slots receive
no fabricated error; independent contexts retain their serialized state. Only
KV exhaustion can trigger a smaller-batch retry, and graph aborts and allocation
errors have distinct messages. A failed system-prefix decode also stops request
processing. Queued error responses now destroy their derived resources safely;
the sanitizer check reproduced the previous invalid base-pointer deletion.

Five generated-model recovery cases pass, including early/later graph aborts
and a system-prefix abort. Six final HTTP controls pass: four real graph aborts
and two simulated status-classification checks. Three focused regressions pass
in both Release and CPU ASAN/UBSAN with leak checking. This establishes scoped
CPU server recovery, not Qwen serving or actual driver-failure recovery. After
the CPU copy-layout repair, all seven HTTP controls pass, including an exact
smaller-batch capacity retry and an unchanged idle-slot saved prefix.

The 2080 SUPER also executes the original model's three-round short-prefix
control with CPU experts: all 2234880 baseline logits pass the fixed reference
bound, all 432 ordered expert selections match, and 18 reset replays are exact.
Its all-API/leak CUDA check reports zero errors/leaks and matches the native
baseline logits/routes byte-for-byte.
This is scoped Turing execution coverage; the full-model numerical gate remains
closed.

A separate single-RTX-3080 control also passes all 2,234,880 short-prefix logits
and 432 ordered expert selections under the explicit F32/F32-KV policy.
Moving five or nine experts per route to the GPU with a 512 MiB expert cache
preserves the CPU control's captured logits and routes byte-for-byte, including
exact native reset replays. These nine-token placement checks do not establish
equal-memory performance, generated throughput or longer-prefix equivalence.

The single-RTX-3080 placement diagnostic also covers the retained 33-token
input. Nine GPU experts per route and all-GPU expert computation preserve all
8,194,560 CPU-control logits and route IDs exactly, with exact reset replays.
The strict source-reference numerical check still fails for 30 full-reference
logits and 31 cached-reference logits; every top-token choice and ordered
expert list matches. This establishes consistency between these placements
for this input, while the normal Qwen architecture gate remains closed.

The full-GPU 33-token RTX-3080, RTX-3060-Ti and RTX-2080-SUPER controls also
pass CUDA memcheck with all API
errors reported and full leak checking: 99 decoded tokens, exact reset replays,
zero errors and zero bytes leaked. Captured logits and routes match the native
control byte-for-byte, and tracked inventory and leases pass their checks.
This does not establish complete physical-memory accounting or a combined cap.

Native RTX-3060-Ti and RTX-2080-SUPER controls also preserve captured CPU-expert
logits and route IDs exactly when moving experts to the GPU. Their strict source
failure counts remain 408/311/309 and 1/1/1; the normal gate remains closed.
The all-GPU path is faster in the 3060-Ti diagnostic and slower in the 2080-SUPER
diagnostic. These short timings do not establish generated throughput.

The same GPU split with a 1 GiB RAM expert cache records 38.5% component hits
and 38.4% fewer logical reads than its 64 MiB control. Outputs remain exact,
but timing does not improve in that run. Cache-hit improvements alone do not
establish faster responses, and these read counters do not measure SSD traffic.

Shared buffers are counted once within an inventory. Mapped model spans describe
address space, not resident RAM. The inventory is not a complete measurement of
process RSS or driver memory, and separate context inventories can share model
storage. Summing them blindly can double-count that storage.

### PLE storage accounts for more than payload bytes

Per-Layer Embedding (PLE) is a large lookup table used by the experimental Qwen
path. Its storage accounting now includes cache backing and indexes, staging
ownership controls, reader queues and tickets, provider metadata, returned
vectors and sequence-history allocations.

Allocation ledgers remain valid when results are moved, copied or retained after
their provider is destroyed. Copy and assignment charge the correct owner;
failed preparation preserves the destination. Reader publication rolls back
partially prepared queue entries and ownership records. Worker-side allocation
failures complete requests with errors instead of leaving them stranded.

These ledgers improve quota enforcement and diagnosis. They do not amount to a
complete aggregate limit on every allocation in a running context.

### Handoff arenas grow transactionally

Multi-backend execution uses private handoff buffers to move intermediate data.
Their replacement is prepared across all affected backends before existing
arenas or tensor bindings change. If preparation fails, candidates are released
and existing arenas remain available.

Per-backend statistics expose live capacity, transient peak, quota rejections
and allocation failures. Growth budgets include both the old arena and its
replacement. Reusing an arena adds no new global synchronization barrier;
retiring one waits for outstanding work.

Component tests cover all three pairs of the RTX 3060 Ti, RTX 2080 SUPER and
RTX 3080, with exact transfer/retry results and zero reported API errors or
leaks. This is component evidence, not whole-model qualification.

### Graph arena growth preserves the committed layout

Graph reservation prepares replacement arenas and layout metadata before
committing them. A failed device allocation or quota admission releases only
the candidates, preserving the previously committed arenas and layout.
Scheduler allocation now propagates a failed reservation.

Identical buffer types share one arena and use the strictest quota among their
aliases. Queries report the last measured requirement, including rejected
requests, and track the old-plus-new allocation peak. The existing size query
continues to exclude duplicate aliases.

The current generation passes 24 distinct Release and 17 CPU
AddressSanitizer/UndefinedBehaviorSanitizer regressions with leak checking.
A reconstruction of the old reservation code fails the ownership test as
intended. Independent/shared graph arenas pass on all three GPUs, and the
handoff tests pass for all three GPU pairs. Both suites report zero errors and
leaks under all-API GPU memory checks. Host metadata allocation failure still
uses GGML assertions; this is not full host-OOM recovery.

### Smaller working chunks when a causal prompt reservation cannot fit

Context initialization now checks graph-memory budgets before reservation and
can reduce a causal microbatch when the original request cannot fit. The
effective chunk is retained by the context and reported in its logs/native
getter. Non-causal inputs and detected encoder contexts retain their full-batch
requirement. An impossible budget returns initialization failure.

The native `compute_buffer_budget_bytes` parameter caps graph arenas per backend;
its automatic CUDA setting respects available memory and the configured reserve.
This is a graph cap rather than a total-context RAM/VRAM limit. The change also
fixes CPU-only split-mode loading that accessed an empty GPU-device list.

Small-model tests exercise a requested 1024-token microbatch, admission at 64,
and a real 128-token prompt. CPU and all three isolated GPUs pass the unchanged
output bound. The GPU API/memory/leak checks report zero errors and leaks.
The original Qwen model's large-batch and whole-device peak-memory acceptance
remain unqualified.

## Session state and speculative verification

The experimental Qwen path includes repairs for shared-sequence ownership,
position ordering, checkpoint restore, transformed journals, defragmentation
and cache retiering.

- Copy-on-write protects state shared by multiple sequences.
- Position handling distinguishes token order from rotary positions and
  preserves the chronological last output during sequence transforms.
- Malformed restore inputs preserve valid output/checkpoint ownership when
  recovery rolls back; prompt tokens are committed only after successful restore.
- Defragmentation preserves recurrent source controls and the actual value-cache
  layout.
- Retiering prepares replacement storage, preserves required checkpoint banks
  and rejects incompatible requests before publication.

Speculative verification also preserves the target sampler's actual verified
state. The old recovery helper could rewind random-number state and replay
acceptance without replaying the sampling operations that changed it. The
repaired helper retains RNG and adaptive sampler state through accepted-prefix
recovery.

Controlled model-backed checks cover rejection and acceptance cases, output
IDs, complete recovered state, logits and subsequent sampler draws. They do not
establish a production drafting system or an accepted-token throughput gain.

## CPU and CUDA execution

### Precision and dispatch repairs

The candidate contains fixes for explicit floating-point precision requests,
quantized projection arithmetic, recurrent input/layout handling, sparse
selection and graph/device ownership. Qualification is recorded per operator,
format, layout and device rather than inferred from one successful model run.

Explicit expert arithmetic covers IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS and
Q2_0 in the qualified CPU and CUDA paths. Indexed and fused gate/up execution
checks routing IDs, pitches, padding and source buffers. Precision-preserving
paths can cost more computation; their existence is not a speed claim.

CPU F32 transpose copies now check the destination layout before using a packed
fast path. Strided KV-cache rows previously received misplaced values and
overwritten padding. All 640 operator cases preserve their destination values
and gaps, and the old-runtime control fails at the first corrupted gap. The
generated model's full and split prompts now produce byte-identical logits.
Twelve focused Release and CPU ASAN/UBSAN/leak regressions pass. This is a copy
correctness repair; it makes no end-to-end speed claim.

### Shape-specific projection tuning

The experimental `ESE_CUDA_EXPLICIT_AUTOTUNE=1` tuner measures eligible small
projection shapes on the active GPU and selects among compiled kernel variants.
Candidates must reproduce the reference arithmetic bit-for-bit before timing
can admit them. Calibration runs outside graph capture, uses bounded scratch
and retains the reference choice when preparation fails.

This is tuning for a qualified projection family. It does not compile an entire
loaded model or provide an FPGA implementation. An independent runtime-compiled
native-CUBIN experiment is separate from the installed tuner.

### Experimental BF16 batching

`ESE_CUDA_BF16_STABLE_BATCH=1` batches qualified BF16 projections while matching
the scalar path. Its bounded, per-backend plan cache is prepared outside CUDA
graph capture. The admitted family uses contiguous, unbiased weights, width
2560, 2560 or 10240 output rows and 2–8 queries. Other shapes retain their
existing dispatch; the default remains disabled.

Isolated projection measurements exclude expert loading, storage, routing and
CPU work. They must not be presented as whole-model speedups. Some mixed-device
vendor-library initialization runs remain unqualified.

### Resource reporting follows the selected GPU

Live resource snapshots now report the CUDA device selected by the model.
Previously, selecting GPU 2 alone reported device 0 and caused the server's
context-resize requests to reject. Selected-device checks pass, and paired
HTTP controls verify shrinking and regrowing the context on GPU 2 with exact
output and token probabilities. Complete physical memory admission remains
unqualified. See [validation](CUDA_SELECTED_BACKEND_VALIDATION.md#resource-snapshots-preserve-physical-cuda-device-ids).

GPU resize preparation also includes tracked live weights, graph/workspace
backing and other fixed device buffers when they exceed the original forecast.
The owner thread rechecks those limits before allocating replacements. Focused
Release/CPU sanitizer and selected Ampere memory checks pass; paired GPU server
controls preserve output and probabilities. This does not establish a complete
RAM/VRAM cap. See [validation](RESOURCE_BUFFER_INVENTORY_VALIDATION.md#server-preparation-uses-tracked-accelerator-floors--2026-10-13).

RAM resize checks now also include tracked live host buffers and metadata
beyond their existing reservations. Mapped weight spans are excluded from
allocated RAM inputs, and reserved PLE cache bytes are counted once. Paired
CPU controls reject an oversized replacement while preserving a fitting live
session and exact output; GPU plans include their host buffers too. Complete
startup and transaction memory admission remain unqualified.

Supported KV replacements now reserve native backend storage, tracked prepared
metadata and bounded migration scratch. The owner thread rechecks the sizing
receipt and audits the prepared transaction before publication. CPU/GPU controls
and cache-only original-metadata tests through 64K match actual allocations;
complete physical memory and 64K model execution remain unqualified. See
[validation](KV_PREPARATION_MEMORY_VALIDATION.md).

Startup planning also reserves the native computation graph metadata arena.
Low-budget CPU starts now reject before context creation, and runtime resize
accounting counts the reservation once. Other startup owners and complete
physical limits remain unqualified. See
[validation](CONTEXT_GRAPH_METADATA_VALIDATION.md).

Contexts with an explicit RAM budget also check native KV, initial output and
tracked host owners before initial buffer allocation. CPU graph backing uses the
remaining quota with microbatch backoff; output replacements charge retained
and incoming buffers and preserve valid state on rejection. Startup, resize and
GPU controls pass with unchanged output. These checks cover tracked storage of
one model/context; complete physical and concurrent memory limits remain
unqualified. See [validation](CONTEXT_INITIAL_MEMORY_VALIDATION.md).

Direct native KV resize and retier calls now apply the context's host limit to
the replacement storage, transaction metadata and migration staging too.
Exact byte boundary tests cover resize, retier and no-op transactions with
retained state preserved on rejection. See
[validation](CONTEXT_KV_HOST_ADMISSION_VALIDATION.md).

## Chat templates

Chat-template integer arithmetic now preserves exact 64-bit values and rejects
overflow or zero modulo with an error. Large finite float values no longer
trigger an unsafe integer conversion. Existing template checks and new boundary
cases pass in Release and CPU sanitizer builds. See
[regression audit](REGRESSION_AUDIT_20261013.md).

## Packaging consistency

### GPU planning follows CUDA visibility and identity

The launcher now matches NVIDIA memory telemetry to CUDA-visible UUIDs.
Automatic split weights follow CUDA's actual device order, and hidden cards
are excluded from available-memory estimates. Calibration identity includes
the ordered CUDA UUID list, preventing reuse after a device binding changes.
The 12-case Linux visibility audit and all 53 launcher/profile tests pass.
CPU runtime planning remains independent of CUDA discovery. See
[device identity validation](CUDA_DEVICE_IDENTITY_VALIDATION_20261013.md).

### Explicit GPU placement is applied before planning

Native device selection, main GPU, tensor split and fixed GPU-layer overrides
now participate in launcher budgets and calibration identity. Single-device
native planning charges the actual selected GPU, and automatic fitting keeps
the requested main device. Three reproduced completion crashes are repaired.
Thirteen release server controls retain saved output exactly; twelve also
preserve it through cache shrink/regrowth. GPU startup can still allocate on
other visible cards. See [placement validation](GPU_PLACEMENT_OVERRIDE_VALIDATION_20261013.md).

Server response flags, token counters and timing values now have defined
defaults. Poisoned-storage direct/queued-copy regressions and strict sanitizer
checks pass. Instrumented resize controls retain exact output within the same
build. The October 14 Linux rebuild includes these changes and the accepted GPU
decode router-submission boundary. Its native payloads match the qualified
source/runtime generation after the declared relocation.

Launcher and Studio version fields now agree on `0.2.1`. Linux CPU runtime
staging uses bundled native libraries and system libraries without retaining
build or Conda paths. Release asset collection requires the requested version
and its matching signature, preventing stale cached installers from being
renamed as the current release.

Current unsigned Linux CPU and CUDA DEB/RPM/AppImage candidates pass loader,
launcher, small-model API/cache and desktop startup checks. CUDA candidates
bundle byte-identical cudart, cuBLAS and NCCL runtime libraries with their
notices; they require a host NVIDIA driver. CPU candidates work without that
driver. Python and OpenMP dependencies are now explicit. Actual DEB installation
passes on Ubuntu 24.04 and actual RPM installation passes on Fedora 44.

Earlier compressed RPM candidates had incorrect raw-payload checksums despite
successful extraction. Rebuilt RPMs use uncompressed payloads and pass both
payload digest checks. These candidates supersede the earlier RPMs.

All 20 packaged resident controls and 20 cache shrink/regrow controls retain
saved content and all 32 probabilities exactly. Eight non-root desktop startup
captures show a ready runtime and optional sharing OFF. These are generated
small-model checks. The October 14 private Windows CI also builds unsigned MSI/NSIS installers
with a separate CPU runtime. Actual CPU executable and automatic launcher
selection checks pass. Both extracted installers match all eight CUDA and five
CPU native files, the standalone launcher and required licenses. Windows GPU
inference, installer execution, signing and release-wide readiness remain
unqualified. See
[Linux CPU/CUDA packaging validation](LINUX_CUDA_PACKAGING_VALIDATION_20261013.md).

The NCCL communicator shutdown leak is repaired: all three GPUs pass memory checks
with all GPUs visible and unchanged primitive-control reports. An optional
registered CUDA shutdown regression covers this path. See
[NCCL cleanup validation](CUDA_NCCL_CLEANUP_VALIDATION_20261013.md).

### CUDA discovery no longer creates unused NCCL ranks

Hardware enumeration now avoids eager all-card NCCL allocation. Communicators
are initialized once when an eligible reduction needs them, and grouped calls
are serialized. Single-device native memory queries avoid other selected-list
rows. Two-participant reductions with three visible cards use the existing
fallback instead of asserting. Backend destruction also frees retained
reduction scratch; the registered three-card memcheck passes with zero leaks.
Thirteen release server controls retain saved output exactly. Pinned-host and
other driver allocations remain outside complete device-isolation qualification.
See [startup validation](NCCL_LAZY_INITIALIZATION_VALIDATION_20261013.md).

## DeepSeek HC normalization

CPU and CUDA hyper-connection normalization now add epsilon after completing
each reduction, matching the pinned reference. The initialized ten-token
reference passes with its original tolerance. HC expansion now retains FP64
combination and post coefficients through the residual expression, resolving
the initialized block's BF16 rounding boundary. Complete CPU Release, CUDA
Release and strict CPU sanitizer suites pass 92/92, 100/100 and 92/92 tests.
The block and token fixtures pass on all three GPUs with the original
references and tolerances; precise expansion stays on the GPU and each card's
primitive memory check reports zero errors and leaked bytes. This is initialized
small-model evidence. See [normalization validation](DEEPSEEK_HC_NORMALIZATION_VALIDATION.md)
and [expansion validation](DEEPSEEK_HC_EXPANSION_VALIDATION_20261013.md).

## Experimental Qwen capability and limitations

CPU convolution fusion now respects the boundaries of the current graph.
This fixes terminal-graph crashes and prevents a short graph view from
executing a later SiLU early while leaving its raw convolution output unwritten.
All 360 component cases and sixteen focused checks pass in Release and CPU
sanitizer builds. Original-model CPU logits and expert routes are unchanged
in the 33-token before/after control.

CPU sum-and-divide fusion now preserves the ordinary operators' reduction axis,
strides, division behavior and sum output. This fixes incorrect normalization
for negative sums and strided tensors, and removes the CPU model's numerical
change when tracing is enabled. The 240-case component regression and focused
Release/CPU sanitizer checks pass. CPU routing weights and logits can change
where the old shortcut used different arithmetic.

Native Qwen cache and layer-placement estimates now include the persistent
sparse-attention indexer keys. The original model's omitted F16 indexer storage
is 192 MiB at 64K capacity. Synthetic and original-metadata size checks pass in
Release and CPU sanitizer builds; actual empty CPU contexts at 4K and 64K
capacity match their cache tensor-payload estimates exactly. Filled-context
operation and complete memory admission remain unqualified.

The native resource controller now estimates each candidate cache capacity from
model metadata, preserving fixed recurrent state instead of scaling a 4K sample.
Runtime resize uses the same sizing data and checks host caches, KV and workspace
against their combined RAM budget. Eighteen focused checks pass in each build;
CPU HTTP shrink/regrowth preserves output and probabilities. Original-model
64K-capacity startup matches its cache payload, and the 33-token CPU control
preserves all logits and expert routes. This does not qualify filled 64K
execution or complete physical memory admission.

The RAM forecast also excludes the Qwen PLE lookup table that the loader streams
from storage. Its 28.8 GB source extent was previously counted as resident dense
weights alongside the separately reserved PLE cache. The corrected original-model
forecast falls from 33.25 GB to 4.45 GB, while preserving cache reservations.
Paired CPU initialization controls reproduce the old 12 GiB budget rejection
and verify current admission with 64 MiB or explicitly requested 1 GiB expert
caches; a 4 GiB budget still rejects. Nineteen focused checks pass per build,
and the original 33-token CPU logits and routes remain byte-identical.
This corrects admission metadata; it does not establish a physical RAM cap,
general cache-reuse gains or a speedup. In a separate 33-token CPU diagnostic,
raising the explicit expert cache from 64 MiB to 1 GiB preserves logits/routes,
increases component-cache hits from 0% to 38.8% and reduces logical storage-read
bytes by 38.7%. This is one workload, without matched repeated timing or
physical SSD-read measurement.

Cache replacement now includes the separate RAM-cache reservation in its
temporary memory peak. A resize can reject when its final cache would fit but
preparing the new cache alongside the old one would cross the RAM limit or host
reserve. Dry-run and committed requests apply the same check; rejected requests
preserve usable resource state and output. Preparation reports expose the
combined RAM peak and headroom. Paired HTTP controls reproduce the previous
over-budget admission and verify rejection plus valid shrink/regrowth in both
Release and CPU sanitizer builds.

The Qwen4Exp text path includes metadata admission and required tensor-shape
checks, PLE lookup, recurrent layers, sparse attention and expert execution. Normal loading
remains gated because the complete release correctness requirements have not
passed.

New numerical reference records identify their Python, PyTorch and NumPy runtime,
thread count and Float32 matmul precision alongside the pinned source and
weights. This helps distinguish backend replay differences from engine errors.
Existing goldens and comparison bounds remain unchanged.

Reference generation also keeps the final output mixer on the chosen decoder
chunk boundaries. Previously, cached decoder runs mixed all final outputs in
one batch, introducing small rounding differences. Six runner regressions cover
single-token and partial chunks alongside the existing checks; all 22 Python
reference/trace tests pass. Original-model head replay verifies the retained
outputs exactly and measures the corrected chunking separately. It leaves the
longer-prompt numerical gate unresolved and preserves the existing goldens.
A complete 48-layer cached rerun confirms that all 543 earlier intermediate
files and weight-access records remain byte-identical; only the final mixer
and logits change as measured by the independent head replay.

Metadata admission now rejects nonfinite or nonpositive RMS epsilon and PLE
convolution kernels with no history. Six paired old/current controls reproduce
the previous acceptance and verify ordinary rejection. Valid metadata still
loads in both Release and CPU sanitizer builds.

The PLE head-table count check widens its multiplication before evaluating the
expected number of offsets. An oversized count that previously wrapped to a
valid-looking value is now rejected. The permanent regression and targeted
Release/CPU sanitizer metadata controls pass with the production gate closed.

Tensor-only checks account for all 1,224 original-artifact tensors across 48
layers. Release and strict CPU sanitizer controls reject 239 in-memory metadata
faults across 47 tensor families, with identical results. These checks load no
weight payloads and do not establish inference parity.
The same 239 missing, dimension and surplus cases also pass as serialized
GGUF faults in both builds. Sparse metadata fixtures preserve quantized block
alignment and valid payload bounds; restored positive controls reproduce the
initial output exactly. No original weight payloads are copied or loaded.

Short original-weight reference checks pass the existing bound
`2e-5 * (1 + abs(reference))`. Code and multilingual prompts each pass 17-token
comparisons, including logits, ordered routes and saved state. The 33-token
check still fails that bound on a small number of logits. Its failing evidence
and original tolerance are retained.

A fresh nine-token F32/non-Flash prefill also passes logits and ordered routes
against the pinned reference and sequential controls. Its selected Ampere-pair
CUDA error/leak check passes with byte-exact native/instrumented output.

Cache gather handling covers F16, F32, BF16, Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0
in qualified layouts. Unsupported configurations are rejected early.
Quantized value caches retain their Flash Attention requirement. Passing a
codec or restore check does not establish its model-quality tradeoff.

Known limitations of this candidate include:

- No complete 64K-context or sustained concurrent-session qualification.
- No complete aggregate memory budget. Actual original-model prefill qualifies
  automatic 1024-to-128 admission for one 128-token pressure workload after
  accounting for known CUDA matrix workspaces. Its sampled physical peaks and
  same-chunk output consistency do not establish filled 64K-context acceptance
  or coverage of every runtime workspace. Current settings differ from the
  historical failed profile.
- Whole-model validation is narrower than GPU component coverage. Both the
  selected Ampere pair and Turing have short-prefix model sanitizer evidence;
  broader placement and long-context checks remain unqualified.
- No Qwen vision or production MTP capability qualification.
- Historical resident-versus-cache output differences remain documented in
  earlier fixtures; before/after cache equivalence does not close those gates.
- No new Qwen serving-slot, readiness or dual-protocol API qualification.

## Performance reporting

The launch presentation now leads with using the PC you already own. Its
[PC guide](WHAT_CAN_MY_PC_RUN.md) separates estimates from tested configurations,
and its [flagship demo record](LAUNCH_DEMO.md) requires actual 64K prompt fill,
sustained generation and measured RAM/VRAM before establishing a headline.

A separate private Qwen3.8 Flash Next GSQ-RCO IQ2_XS benchmark on the RTX 3060 Ti
and RTX 3080 records 6.7273 seconds for ESE and 1.2213 seconds for Strata to
complete 32 warm output tokens, using the same raw prompt/count and fresh
prompt/session state. ESE is about 5.51 times slower for those configurations.
Policies and generated continuations differ; OS page-cache state is uncontrolled.
Both immutable CPU references pass the selected model’s 33-token numerical
control, but it has no filled-64K qualification. These results do not resolve the
earlier original-model 33-token failure or enable the production Qwen gate.
They are an experimental diagnostic, not the supported launch demonstration.


There is no current matched release benchmark proving an ESE-versus-Strata
speed ratio. Historical diagnostic runs used different speculation, cache,
context and batch settings, so they do not support a "10×" claim.

Cache improvements, worker counts and specialized kernels are evaluated against
the same model and resource limits. Individual historical timings and component
measurements remain scoped to their recorded workloads. This draft deliberately
does not turn them into a release headline.

## Compatibility and validation record

The name remains **ESE — Expert Streaming Engine**. Experimental options above
do not automatically enable normal Qwen loading. Source users must rebuild the
runtime and callers together where parameter structures have changed; a new
server should not be mixed with stale libraries.

Current evidence is distributed across focused validation records. Test counts
describe their own source generations and should not be summed as a single
release certification:

| Area | Evidence | Scope |
| --- | --- | --- |
| Expert-cache concurrency | [Validation](CACHE_CONCURRENCY_VALIDATION.md) | Controlled concurrency, bounded staging, specific A/B model checks |
| Payload retention | [Validation](CACHE_PAYLOAD_RETENTION_VALIDATION.md) | Layout/padding ownership and specific exact model comparisons |
| Load queue | [Implementation and checks](EXPERT_LOAD_QUEUE.md) | Opt-in bounded worker path |
| Memory inventory | [Validation](RESOURCE_BUFFER_INVENTORY_VALIDATION.md) | Tracked native buffers; not total RSS |
| PLE allocation ownership | [Vector checks](PLE_VECTOR_ACCOUNTING_VALIDATION.md), [history checks](HISTORY_HOST_ALLOCATION_VALIDATION.md) | Quotas, copies, rollback and lifecycle |
| Handoff arenas | [Validation](HANDOFF_ALLOCATION_VALIDATION.md) | CPU failure injection and all three GPU pairs |
| Graph arenas | [Validation](GRAPH_ARENA_BUDGET_VALIDATION.md) | 24 Release/17 CPU sanitizer tests; all three GPUs; zero GPU errors/leaks |
| Prefill backoff | [Validation](PREFILL_BACKOFF_VALIDATION.md) | 25 Release/18 CPU sanitizer tests; small-model 1024-to-64 admission and 128-token prompt; all three isolated GPUs |
| Original Qwen initialization | [Validation](QWEN_PREFILL_INIT_VALIDATION.md) | Native pressure backs off 1024-to-256; instrumented 1024-to-128, zero errors/leaks; empty context only |
| CUDA workspace pools | [Validation](CUDA_POOL_INVENTORY_VALIDATION.md) | 26 Release/18 CPU sanitizer regressions; VMM and legacy projections/inventory on three GPUs, six zero-error/leak memchecks |
| CUDA pool admission | [Validation](CUDA_POOL_ADMISSION_VALIDATION.md) | Cold/growth/partial rejection and exact retry, live leases/overflow; 14 final zero-error/leak memcheck processes |
| Shared compute admission | [Validation](SHARED_COMPUTE_BUDGET_VALIDATION.md) | 27 Release/19 CPU sanitizer regressions; graph/pool mutual admission and 15 final zero-error/leak GPU memchecks |
| Original Qwen prefill | [Validation](QWEN_PREFILL_EXECUTION_VALIDATION.md) | Nine-token F32 logits/routes and clean Ampere-pair memcheck; retained arena-only pressure failure and explicit-128 resource control |
| Workspace-aware prefill | [Validation](CUDA_WORKSPACE_PREFILL_VALIDATION.md) | Automatic 1024-to-128 pressure admission and actual 128-token execution; byte-exact controls and clean Ampere-pair memcheck; fresh F32 nine-token logits/routes; six clean VMM/legacy component memchecks on three GPUs |
| Persistent decode resources | [Validation](QWEN_PERSISTENT_DECODE_VALIDATION.md) | 30-minute repeated nine-token workload; 1692 exact replays, stable tracked inventory and ownership drain; selected Ampere pair |
| Decode abort status | [Validation](DECODE_ABORT_STATUS_VALIDATION.md) | Distinct abort/KV-capacity results, ten actual-Qwen accepted-prefix recoveries, old-runtime negative control; 28 Release/20 CPU sanitizer regressions |
| Server decode recovery | [Validation](SERVER_DECODE_RECOVERY_VALIDATION.md) | Five generated-model abort/recovery cases; derived response ownership fixed; seven HTTP controls pass with the validated CPU copy guard |
| CPU transpose copy | [Validation](CPU_TRANSPOSE_COPY_VALIDATION.md) | 640 destination-layout cases, old-runtime negative control, exact model full/split logits, seven HTTP controls and twelve Release/CPU sanitizer regressions |
| CPU normalization fusion | [Validation](CPU_SUM_DIV_FUSION_VALIDATION.md) | 240 complete/staged graph cases; six old-library controls; exact CPU model traced/untraced logits/routes; thirteen regressions plus loader-gate check per build |
| CPU convolution graph boundaries | [Validation](CPU_SSM_CONV_BOUNDARY_VALIDATION.md) | Terminal/prefix negative controls; 360 scalar/history/boundary cases and sixteen checks per build; original CPU model logits/routes unchanged |
| Turing model execution | [Validation](QWEN_TURING_MODEL_VALIDATION.md) | Original-model nine-token baseline passes fixed bound/all ordered expert routes; 18 exact reset replays with CPU experts |
| Sampler recovery | [Validation](SPECULATIVE_SAMPLER_RECOVERY_VALIDATION.md) | Controlled acceptance/rejection and RNG/state recovery |
| Explicit projection tuning | [Validation](CUDA_EXPLICIT_PROJECTION_TUNING.md) | Experimental eligible projection shapes |
| BF16 batching | [Validation](CUDA_BF16_STABLE_BATCH_VALIDATION.md) | Experimental shape/device-specific family |
| Qwen reference coverage | [Validation](QWEN4EXP_PROMPT_COVERAGE_VALIDATION.md) | Short prompts pass; original 33-token gate fails |
| Qwen metadata admission | [Validation](QWEN4EXP_LOADER_GEOMETRY_VALIDATION.md) | 45 malformed metadata cases; six paired old/current controls; all 1,224 tensors accounted for; 239 in-memory and 239 serialized tensor faults rejected per Release/CPU sanitizer build; integrated head-count check passes four focused CTests and 18 targeted controls |
| Qwen cache formats | [Validation](QWEN4EXP_CACHE_FORMAT_VALIDATION.md) | Gather/codec and specific lifecycle checks |
| Qwen cache sizing | [Validation](QWEN4EXP_CACHE_SIZING_VALIDATION.md) | 540 synthetic and 144 original-metadata cases per build; twelve focused checks; actual CPU 4K/64K-capacity initialization matches tensor payloads |
| Native cache forecasts | [Validation](NATIVE_CACHE_FORECAST_VALIDATION.md) | Fixed/stepped state preserved; eighteen checks per build; CPU HTTP shrink/regrowth exact; original CPU initialization forecast exact and 33-token outputs unchanged |
| Streamed PLE RAM forecasts | [Validation](PLE_RESIDENT_FORECAST_VALIDATION.md) | Old/current budget controls; nineteen checks per build; original 4K-capacity CPU initialization and unchanged 33-token logits/routes |
| Host replacement peaks | [Validation](HOST_CACHE_PREPARATION_VALIDATION.md) | Combined RAM cache/KV/workspace admission; four old planner controls; paired HTTP over-budget rejection and valid shrink/regrowth; eighteen checks per build |
| Native KV preparation | [Validation](KV_PREPARATION_MEMORY_VALIDATION.md) | Five Release/four strict CPU checks; 22 HTTP controls; exact original-metadata cache allocation/migration/rollback through 64K; three Ampere-pair memchecks with zero errors/leaks |
| Graph host metadata | [Validation](CONTEXT_GRAPH_METADATA_VALIDATION.md) | Six Release/five strict CPU checks; 12 startup/API and six GPU HTTP controls; original-metadata forecasts; two zero-error/leak Ampere memchecks |
| Native context host admission | [Validation](CONTEXT_INITIAL_MEMORY_VALIDATION.md) | Six Release/five strict CPU checks; startup boundary rejection, graph backoff and retained output growth; 12 CPU/six GPU HTTP controls; two zero-error/leak Ampere memchecks |
| Native KV host admission | [Validation](CONTEXT_KV_HOST_ADMISSION_VALIDATION.md) | Nine native byte boundary cases; old-library negative; exact retained state/receipts; six Release/five strict CPU checks; CPU/GPU controls and two clean memchecks |
