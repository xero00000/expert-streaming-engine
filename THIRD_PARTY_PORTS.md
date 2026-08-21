# Third-party port provenance

ESE is MIT licensed and derives from `ik_llama.cpp` / `llama.cpp`. Selective external ports must preserve their own provenance and license notices.

## buun-llama-cpp — Turbo KV Phase 1

- Source repository: `spiritbuun/buun-llama-cpp`
- Source commit: `799e3995cd4f19aa9f6a3fa9fb5b4674422bf0ee`
- Source tree: `a2bd802d81936bab8a066cbf789a427776fb4839`
- Source file: `ggml/src/ggml-turbo-quant.c`
- Source license: MIT
- ESE adaptation:
  - `ggml/include/ggml-turbo-kv.h`
  - `ggml/src/ggml-turbo-kv.cpp`
  - `tests/test-turbo-kv.cpp`
  - `scripts/test-turbo-kv.sh`
  - `docs/TURBO_KV_PHASE1.md`

Adapted elements:

- deterministic seed-42 Gaussian rotation;
- modified Gram-Schmidt construction;
- Turbo4 and Turbo8 Lloyd-Max centroid tables;
- norm extraction and reconstructed-norm correction;
- 4-bit nibble packing and 8-bit index storage;
- inverse-rotation decode.

ESE-specific changes:

- checked standalone C ABI;
- deterministic thread-safe immutable initialization;
- explicit size/alignment/overflow validation;
- non-finite input rejection;
- no registration as a public GGML/KV type yet;
- no import of incomplete CPU stub paths for Turbo2, Turbo3, or TCQ.

No code from this source is represented as an original ESE invention. Subsequent ports must append their exact source commit and file list here.
