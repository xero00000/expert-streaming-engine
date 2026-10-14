# Native KV preparation respects context host limits

Native resize and retier preparation now use the host limits attached during
context initialization. Before physical replacement allocation, the native
receipt charges incoming host KV storage, transaction/cache metadata and bounded
migration staging while retaining current storage. This also charges metadata
for a no-op transaction. Actual prepared capacity is checked before allocating
migration payloads, and retained staging capacity is checked before migration.

This extends the native context checks to direct API callers. The server retains
its separate owner receipt and publication audits. No API/structure layout,
model arithmetic, weight, reference or tolerance changes are included.

## Evidence

Evidence: `/home/xero/Desktop/test/results/ese-context-kv-host-admission-20261013`.

- A standalone probe linked against the frozen previous binary reproduces a
  direct shrink transaction accepted at a 29,931,012 B host limit with
  30,058,956 B tracked retained storage, an excess of 127,944 B. Freeing the
  transaction restores the original inventory.
- The new permanent boundary test fails against that frozen library with
  `host KV quota accepted preparation one byte over its limit`.
- Six Release and five strict CPU ASan/UBSan/leak CTests pass. Nine native
  cases test one byte below, at and above the preparation peak for resize,
  retier and no-op transactions. The Release fixture's respective incoming
  forecasts are 127,976 B, 441,400 B and 1,136 B. The peak includes temporary
  staging, explaining its difference from retained transaction inventory.
  Accepted actual metadata/staging receipts equal forecasts. Publication and
  rollback retain fitting storage; rejected preparation and completed rollback
  preserve serialized state, logits and allocation counts exactly.
- Twelve final startup/API and six old/current GPU HTTP controls pass. Startup
  checks from the preceding change remain enforced. CUDA2, split CUDA0/2 and
  mixed CPU/CUDA2 resizing retain exact KV receipts, content and all 32 reported
  probabilities. Both builds pass 540 synthetic and 144 canonical original
  metadata sizing cases. Those original cases load no weight buffers and create
  no inference context.
- Two Ampere-pair CUDA memchecks report zero errors and leaked bytes. Owned
  jobs are terminal. Source before/after, the owned patch, frozen binaries and a
  verified manifest retain the result and exclude preliminary runner mistakes
  from qualification.

## Scope

This covers supported contiguous native cache layouts and the tracked storage
of the limited model/context. Admission-query/constructor temporaries,
untracked containers, allocator/backend/vendor/OS overhead, unrelated contexts
and concurrent application owners remain outside comprehensive admission.
The actual-capacity audit is not a measurement of every physical allocation
peak. Complete RAM/VRAM limits, original-model numerical parity, 64K inference,
concurrent serving and release readiness remain unqualified. Normal Qwen
loading stays gated.
