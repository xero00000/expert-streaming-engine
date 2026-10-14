# PLE staging allocation and reservation ownership — 2026-10-11

Async PLE telemetry now reports logical reserved bytes separately from live
data-buffer capacity and shared-buffer control-block allocation requests.
Control blocks remain counted while weak references retain them after data
release. The native inventory reports staged data in `PLE_STAGING` and its
tracked shared controls in `HOST_METADATA`.

Buffer construction previously transferred reservation ownership after
constructing a `shared_ptr`. If shared control allocation failed, the standard
constructor invoked the supplied deleter and released the reservation; later
operation cleanup tried to release it again and terminated the process.

The worker now owns raw data with a unique pointer during allocation/accounting,
publishes the reservation transfer before shared control construction, and uses
the buffer deleter for exactly one data/reservation release. Partial batch
construction and source errors propagate through futures and release storage.
The buffer-vector reserve also moves inside the worker's existing error handler.

An optional control-allocation quota enforces actual rebound allocator request
sizes and enables real control-block allocation failures. Default construction
leaves this metadata quota unlimited; the existing logical staging limit remains.
This quota does not cover all reader bookkeeping or total process memory.

## Verification

Evidence: `/home/xero/Desktop/test/results/ese-ple-staging-ownership-20261011`.

- Zero-control and one-control quotas fail the actual shared ownership
  allocation, including a failure partway through a two-span batch. Reservations,
  data and controls drain; a later single-span read succeeds with the one-control
  quota. Zero-quota retries return the same allocation error without termination.
- A one-worker blocked-source case distinguishes eight reserved bytes from four
  allocated bytes with another request queued. Returned bytes remain exact.
- A weak reference retains a control block after data/reservations drain; its
  final reset releases metadata. The host's measured single control request is
  72 bytes. The fixture derives its quota from the measured value, not that
  platform-specific size.
- Source-read failures across a bounded batch drain storage and permit exact
  retry. Existing deduplication, cancellation, invalid-span and staging-limit
  checks remain passing.
- The native row-store inventory checks data and control allocation while a
  consumed ticket still owns its completed futures, then checks release.
  Cancelled tickets and two-tier temporaries are drained before comparing
  independent arena/index metadata.
- An offside library restoring the old transfer order, with the same new
  allocator/quota, reproduces SIGABRT (-6). The identical current executable
  passes. This reconstructed order control is not an old released binary;
  core dumps are disabled for the owned negative subprocess.
- Final 19 Release and 12 CPU ASAN/UBSAN/leak tests pass, including the shared
  DeepSeek Engram provider. Focused storage/inventory tests pass in both builds.

## Remaining work

This stage covers shared-buffer construction/ownership and tracked staging
data/control requests. It does not qualify every allocation failure in reader
submission, worker preparation or completion bookkeeping. Submission still
allocates queue/index/owner metadata while publishing requests; worker batch
preparation and latency sampling also have allocation sites to harden.

Vector objects, ticket/promise/future and registry metadata, histories, allocator
headers, provider/device-private allocations and complete pre-allocation budgets
remain outside this inventory. It is not total RSS or complete C1/C3 closure.
No new whole-model/GPU numerical, long-context or performance qualification is
claimed. The earlier cache-format GPU/model checks remain a separate generation.
