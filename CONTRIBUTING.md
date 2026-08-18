# Contributing to Expert Streaming Engine

ESE is a performance-oriented inference fork, but correctness and bounded memory come before benchmark wins.

## Before opening a change

1. Base general work on `ese-unified`.
2. Keep Android/QNN work on its platform branch unless the change is backend-neutral.
3. Run:

```bash
./ese doctor
python -m unittest discover -s tests -p "test_*.py" -v
./ese build --backend cpu
```

4. For CUDA changes, record the GPU architecture and prove the intended kernel/path executed.

## Pull-request scope

Prefer one independently testable capability per pull request. Avoid combining:

- upstream synchronization;
- a new model architecture;
- a cache-policy change;
- a new quant/KV codec;
- broad documentation cleanup.

Small, reviewable ports are easier to validate and preserve across future ik/llama synchronization.

## Required evidence

### Lossless changes

Include, where applicable:

- exact token/output parity;
- route ID parity;
- logits or selected intermediate parity;
- one cold and one warm run;
- peak RAM and per-device VRAM;
- proof that no fallback path handled the work.

### Lossy quantization or KV changes

Also include:

- F16 reference;
- bits per value including metadata;
- perplexity;
- KLD distribution, not only a mean;
- at least one task-quality panel;
- prompt-processing and decode speed separately;
- context-depth sweep.

### Cache and storage changes

Also include:

- configured capacity;
- forced eviction;
- hit/miss/eviction counts;
- bytes read and read backend;
- lease/in-flight lifetime test;
- bounded-memory trace;
- cold-cache behavior.

## Performance reports

Provide:

```text
commit
model and quant
complete command/environment
CPU/RAM/storage
GPU/driver/interconnect
context fill
prompt tokens
generated tokens
repetition count
median and low tail
cold or warm state
```

Do not promote a faster result that changes output unexpectedly, relies on an unreported warm cache, or silently falls back to another backend.

## External ports

ESE is MIT licensed, but imported code still requires provenance.

For every port:

- pin the source repository and commit;
- identify the original files/commits;
- retain copyright and license notices;
- separate mechanical adaptation from ESE-specific redesign;
- document behavior differences;
- add acceptance tests before exposing a stable flag.

Do not merge an external fork wholesale. Port the smallest coherent layer that satisfies ESE's architecture.

## Design rules

- Every cache is bounded.
- Every accelerated path is observable.
- Reconfiguration is failure-atomic.
- In-flight tensors cannot be evicted.
- Disk extents use checked 64-bit arithmetic.
- No feature is called complete based on compilation alone.
- Defaults are promoted from repeated workloads, not one best run.
- Native options remain available; the `ese` launcher must print what it selects.

## Documentation

User-facing changes should update one focused document under `docs/` and keep the README limited to the main path. Detailed experiments belong in benchmark records or issues, not in the landing page.
