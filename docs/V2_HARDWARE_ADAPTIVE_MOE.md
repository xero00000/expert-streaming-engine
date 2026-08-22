# ESE v2: hardware-adaptive MoE

This document is the implementation contract for the v2 hardware-adaptive MoE work. Development occurs in the private v2 repository; completed, independently implemented phases can later be reviewed for public `main`.

## Invariants

- GGUF remains the canonical model format.
- The storage hierarchy remains disk to bounded RAM leases to bounded per-device VRAM caches.
- Calibration must exercise ESE/GGML production primitives. Nominal PCIe rates and synthetic-only estimates cannot drive policy.
- FreeToken is an algorithmic reference, not a source-code dependency. No FreeToken engine, Python serving stack, Triton core, or copied Apache-2.0 implementation enters ESE's MIT codebase.
- A missing probe is reported as `unavailable`; it is never silently replaced by a synthetic value.

## Phase A: hardware calibration

The user-facing commands are:

```sh
./ese build
./ese calibrate
./ese hardware-profile
```

Profiles default to `~/.cache/ese/hardware-profile.json`. They are versioned, written atomically with mode `0600`, and bound to a fingerprint covering CPU/NUMA topology, GPU UUID/model/device order, driver, PCI bus identity, CUDA compute capability, kernel, and the NVIDIA topology matrix. A changed identity makes the profile stale.

### Measurement maturity

| Measurement | Current source | Planner-ready |
| --- | --- | --- |
| Sustained host copy | native timed `memcpy` | No; baseline only |
| H2D/D2H | `ggml_backend_tensor_set/get` on a real CUDA backend | No; missing expert geometry/type keys |
| H2D under host-memory contention | CUDA upload concurrent with host copy | No; baseline only |
| CPU MoE | Explicitly `unavailable` | No |
| Adaptive expert-cache upload | Explicitly `unavailable` | No |

The planner must not consume Phase A data until the last two rows use their production paths and results are keyed by expert GGML type, geometry, GPU, and NUMA node.

### Remaining Phase A gate

- Add a model-backed probe for the real ESE CPU MoE kernel.
- Expose a narrow benchmark entry point around the production adaptive expert-cache upload path.
- Run CPU MoE and cache upload concurrently and record both completion times.
- Record per-format/per-geometry samples and confidence statistics.
- Validate numerical output against a CPU reference.
- Feed only complete, current, sufficiently confident profiles into the native resource controller.
- Add controller tests for all-upload, all-CPU, and mixed integer splits, including hysteresis and small miss counts.

## Later phases

1. Phase B: native CPU/GPU hybrid decode co-execution.
2. Phase C: double-buffered prefill streaming.
3. Phase D: live KV/expert/transient resource rebalancing.
4. Phase E: optional aligned FastStore sidecar bound to GGUF identity.
5. Phase F: semantic cache anchors and newer hardware kernels.

Each phase needs correctness tests, local benchmark evidence, documented hardware coverage and limitations, and a clean private review checkpoint before it is proposed for public `main`.
