# Qwen PLE lifecycle conflict and range safety

Evidence: `/home/xero/Desktop/test/results/ese-qwen-sequence-atomic-20261011`.
This is scoped B4 progress; position transforms and full release parity remain open.

PLE sequence copy now prepares and validates the destination before publishing
its tokens or rolling hash. The public cache-copy wrapper also prepares PLE
history before touching KV ownership, checkpoints or saved sequence logits.
A late conflicting position therefore leaves those live records intact.
Malformed PLE history restore (duplicate or negative positions) likewise leaves
the original history intact. This does not establish atomicity of the entire
session reader or arbitrary allocator/backend failures.

Wildcard sequence removal now applies the requested range to each PLE history,
retaining tokens outside that range instead of clearing every history. Empty
or reversed ranges are no-ops in the PLE helper. Rebuilt rolling hashes are
checked independently with the original integer hasher. This does not certify
reconstructing all recurrent tensors after arbitrary interior token removal.

The new conflict regressions fail against the pre-change implementation:
PLE helper reports changed live history; public copy reports changed cache
metadata. After repair, 13 focused Release CTests and two strict CPU
ASAN/UBSAN/leak CTests pass. Additional checkpoint-preservation assertions pass
in Release and the sanitizer build. These tests use synthetic state for the
conflict and wildcard paths and do not open the production model gate.

The real original-weight model lifecycle probe completes with exit zero and
18 exact logit/route comparisons, including sequence copy/remove/keep,
restore/replay, speculative accepted-prefix recovery and session migration.
Baseline nine-token teacher logits, input IDs, routes and both original saved
snapshots remain byte-identical to the previous validated generation. No new
numerical kernels or performance changes are qualified here.

Public-copy preparation currently clones PLE histories, adding temporary host
memory proportional to retained history. Allocation-cost accounting and a
smaller prepared transaction remain C1/C3 work; no bounded-allocation claim
is made for that operation.

Next: implement coherent sequence-position add/divide handling for PLE history
and the output ledger, including stable chronological ordering when division
creates equal positions. The current map stores one token per position and
ordered QSA rejects duplicate positions, so updating the ledger alone would
leave compression broken. Pending work includes persistence/copy/rollback of
that ordering, actual shifted-key checks and interleaved-session failure tests.
The 33-token strict numerical discrepancy, long context, resource work,
kernels/cache/speculation, fair Strata benchmarks, serving, README
rebrand and GitHub publication all remain open.
