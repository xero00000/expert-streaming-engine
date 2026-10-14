# Launcher CUDA device identity validation

Date: 2026-10-13. This qualifies visible-device discovery, automatic split
ordering, and calibration identity in the tested Linux launchers. The release
is still incomplete.

## Failure and repair

The launcher previously used `nvidia-smi` indices directly as CUDA indices.
In the Ubuntu test environment, NVML lists the RTX 3060 Ti, RTX 2080 SUPER,
and RTX 3080 in that order. CUDA's default order is RTX 3080, RTX 3060 Ti,
then RTX 2080 SUPER. The old automatic split therefore assigned the free-memory
weights to the wrong cards. With only the RTX 3080 exposed by UUID, the old
launcher still reported all three cards and their combined memory.

The launcher now asks the CUDA driver for visible-device UUIDs in logical
ordinal order, then joins those UUIDs to NVIDIA memory telemetry. `GPUInfo.index`
is the CUDA ordinal; `nvml_index` and `uuid` retain the physical telemetry
identity. Hidden cards are excluded, and automatic tensor-split weights follow
the visible CUDA order. The driver interprets numeric indices, UUIDs, UUID
prefixes, and invalid-index truncation; the launcher does not duplicate those
rules. No context-creation or GPU allocation API is called by discovery.

The hardware-profile fingerprint now includes the ordered CUDA UUID list.
Changing visible cards or their order invalidates prior calibration bindings.
Equivalent mappings remain valid after a profile's JSON save/load round trip.
Profiles collected before this identity field was added require recalibration.

Unavailable CUDA discovery or unmatched memory telemetry produces an explicit
planning error instead of assuming an index mapping. CPU runtime planning
does not require CUDA discovery. Model arithmetic, references, tolerances,
weights, and the normal Qwen architecture gate are unchanged.

## Qualification

The native probe links the retained Ubuntu CUDA backend and compares its
logical device names and runtime UUIDs with launcher plans and calibration
identity. All 12 cases agree:

| Cases | Result |
| --- | --- |
| Default, explicit `FASTEST_FIRST`, explicit `PCI_BUS_ID` | Native, launcher, and profile UUID order agree |
| Numeric reordered visibility and invalid-index truncation | Same visible subset and order |
| Reordered full UUIDs and abbreviated UUID | Same visible subset and order |
| Empty visibility and an invalid first index | No usable CUDA devices reported |
| Each of the three cards exposed alone by UUID | One CUDA device, numbered 0, with the correct physical identity |

Every automatic tensor split matches the free memory of the corresponding
visible cards. The new permanent ordering test fails against the retained old
launcher in all four synthetic visibility cases.

- All **53** launcher and hardware-profile unit tests pass on the host and in
  Ubuntu Python 3.12. The Windows driver-loader branch is tested with a mock;
  Windows execution is not qualified here.
- All **19** Studio Node tests pass.
- Actual source-launcher sessions select each requested physical GPU UUID,
  confirmed by the owned server PID in NVML telemetry and the native resource
  snapshot. Disabled visibility produces a CPU-only session.
- The old and new launchers return identical content and all 32 saved token
  probabilities on the same RTX 3080.
- Fresh CPU and CUDA runtime staging includes both updated Python files.
  Staged version/marker, doctor, and plan checks pass. Actual staged CPU and
  three staged GPU sessions return all 32 probabilities exactly matching their
  saved CPU or same-device controls and load the staged project libraries.
- Saved/reloaded profiles using the observed CUDA identity reject every
  changed binding and accept equivalent mappings.

## Commands and evidence

Permanent Python controls:

```sh
python3 -m unittest tests.test_ese_launcher tests.test_hardware_profile
```

Evidence is retained under `results/ese-cuda-device-identity-20261013`:
source before/after, old launcher, commands, terminal exits, the native probe,
12-case mapping proof, physical server ownership, calibration binding proof,
staged runtimes, scoped patch, and SHA-256 manifest. Preliminary observer
failures are retained separately from the successful qualification runs.
Prior sealed NCCL and Ubuntu evidence is unchanged.

The implementation follows NVIDIA's
[CUDA visibility and ordering rules](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#env-vars)
and [driver device APIs](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__DEVICE.html).
The execution evidence uses Ubuntu 24.04, CUDA 12.8.61, and the RTX 3060 Ti,
RTX 2080 SUPER, and RTX 3080. MIG devices, other operating systems, and manual
native placement overrides are not qualified by this visibility audit.
Desktop CUDA packages, original numerical gates, full-model/64K/concurrency,
Windows installers, signed updates, and release publication remain open.
