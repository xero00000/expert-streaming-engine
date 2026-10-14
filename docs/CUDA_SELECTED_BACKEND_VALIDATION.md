# Initialize only the model's selected CUDA backends

After the fresh-thread CUDA context repair, an ordinary full-model control
failed during initialization of CUDA1, the unselected RTX 2080 SUPER. CUDA
reported the device busy/unavailable. The model's weights were assigned to
CUDA0 and CUDA2. The context initializer nevertheless created backends for
every visible GPU before filtering the list; unused CUDA backends also lost
their owner when excluded by that filter.

CUDA context creation now iterates the model's selected raw device IDs.
Single-device mode translates the main GPU's local index as before, and
CPU-only selections add no GPU compute backend. RPC IDs beyond the local GPU
range remain for the existing RPC initialization branch. RPC transport and
other accelerator backends are outside this qualification.

Evidence is under
`/home/xero/Desktop/test/results/ese-selected-cuda-backends-20261011`:

- A generated tiny Llama GGUF exercises real model/context creation: CUDA0,
  noncontiguous CUDA0/CUDA2, CPU-only selection and main-index-zero selecting
  raw CUDA2. No external model is required by the regression.
- An interposer makes unselected backend CUDA1 unavailable. The same final
  executable fails against the frozen old library and passes all four
  selections against the candidate. The candidate also passes unsuppressed
  all-API CUDA memcheck with zero errors. The interposer rejects unexpected
  backend construction; it does not suppress CUDA errors or alter kernels.
- Eight normal regressions pass. Strict sanitizer checks also exposed an
  uninitialized split-mode enum in synthetic test models; value-initializing
  the three dummy-model fixtures fixes that fixture defect. All three
  affected normal and ASan/UBSan/leak tests pass after the correction. The
  actual CUDA selection test is qualified by native execution and CUDA
  memcheck, rather than the CPU-only sanitizer build.
- Fresh real-model CPU and GPU-fallback checkpoint controls pass with all
  three GPUs visible and the unselected-backend blocker active. Each passes
  18 JSON lifecycle checks plus three checkpoint controls. Each preserves
  8,194,560 F32 teacher logits, baseline logits and baseline MoE routes
  byte-for-byte against the frozen earlier single-window generation. RAM
  capacity/staging and final lease/load/reservation drain are audited.

The fresh gated library includes the current sequence-bank candidate, split
loader cleanup and Qwen scalar metadata validation. Its ordinary teacher
checks do not establish upstream full-model parity or complete bank recovery.

The first bank diagnostic inherited single-sequence recovery flags and
incorrectly selected one sequence. The probe now gives its explicit bank
mode priority. The corrected test then passed accepted counts zero through
three for target sequence zero, but failed the other sequence's continuation
at accepted count four. Restored complete sequence bytes and owned last logits
matched before continuation. All four host/CUDA graph reuse ablations reproduce identical failing
outputs. Layer traces localize the first numerical divergence to layer 3
attention: query and logical visible key/value inputs are byte-identical,
but their physical cache slots differ. Independent F64 attention agrees
across those logical inputs. The current CUDA attention output depends on
slot placement; its repair and the two-sequence candidate remain unqualified.

No speed, long-context, MTP or production support claim follows from
this change. Failed initialization, diagnostic-mode and fixture attempts are
retained separately from the passing controls.

## Resource snapshots preserve physical CUDA device IDs

The public cache resource snapshot previously returned the scheduler backend
slot as its device ID. Selecting CUDA2 alone creates scheduler backend zero;
its snapshot reported device zero while the resource plan correctly named
device two. Server dry-run and committed context resizing then rejected the
request because the live snapshot appeared to omit the planned accelerator.

`llama_resource_get_snapshot` now translates CUDA backends to their physical
CUDA device IDs. Its public header and non-CUDA behavior are unchanged. The
separate native memory-pool inventory retains its documented scheduler backend
namespace; that inventory must be translated explicitly before planner use.

Evidence: `/home/xero/Desktop/test/results/ese-resource-device-identity-20261013/qualification-final.json`.

- The permanent selected-device regression checks snapshot IDs before and after
  decode across CUDA0, CUDA0/CUDA2, CUDA2 alone and two CPU storage modes. The
  saved old library fails this assertion; the current Release build passes.
- Three focused Release checks and two strict CPU ASan/UBSan/leak checks pass.
  The CPU checks cover planner/retier behavior; they do not execute CUDA mapping.
- Paired HTTP controls on a supported three-layer Llama fixture reproduce the
  old CUDA2 plan/snapshot mismatch and resize rejection, then verify current
  dry-run, 512-to-256 shrink and regrowth to 512. Generated content and all 32
  token probabilities stay exact across old/current and before/after resizing.
  Each server's loaded library is verified from its process mappings.
- Two harness setup failures are retained separately and excluded: the first
  loaded the current library instead of the archived control; the second
  compared KV occupancy before the prompt with occupancy after the prompt.

No additional CUDA memory-sanitizer run was performed for this mapping change.
The production Qwen gate remains closed in both builds. Complete aggregate
memory admission, concurrent context ownership and filled long-context
execution remain unqualified.
