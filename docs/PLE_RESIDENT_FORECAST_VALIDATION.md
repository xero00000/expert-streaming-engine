# Streamed PLE source and resident RAM forecasts

The native planner excludes the skipped Qwen4-Exp PLE lookup table from its
dense-weight RAM forecast. The loader stores that table's validated file extent
and streams rows through a bounded provider. Its full source size previously
remained in the expert index's dense-byte total, while the bounded PLE RAM cache
was also reserved separately.

For the retained original model, the skipped IQ4_NL table contains
28,800,138,240 bytes. The correction reduces the dense-weight forecast from
33,251,397,120 to 4,451,258,880 bytes. Expert source bytes and the separate PLE
cache reservation are preserved. This changes planning metadata, not model
weights, arithmetic or default cache capacities.

The subtraction applies only to a nonempty deferred expert index with a valid
Qwen4-Exp PLE source descriptor. Without that index, the existing sum of
materialized tensors already excludes skipped weights and is retained. Other
architectures keep their previous estimates. An inconsistent PLE extent larger
than the indexed dense total returns an unrepresentable estimate, causing
budget admission to reject rather than underflow.

Evidence: `/home/xero/Desktop/test/results/ese-ple-resident-forecast-20261012`.

- The new resource-byte regression fails against the saved old Release and CPU
  sanitizer libraries. It passes null/optional-output, non-Qwen, absent and
  zero-length source, zero-resident, inconsistent-extent and materialized-tensor
  fallback controls. It also exercises actual cache-plan admission.
- Nineteen focused CTests pass in each Release and strict CPU ASan/UBSan/leak
  build, including resource planning, cache/state lifecycle, server recovery
  and the normal architecture rejection gate.
- Original-model CPU initialization at 4K capacity rejects a 12 GiB RAM budget
  with the old library. The corrected library admits the same requested
  64 MiB expert cache and 256 MiB PLE cache. A separate control retains an
  explicitly requested 1 GiB expert cache with auxiliary RAM cache disabled.
  Both initialized contexts match the native 349,323,264-byte cache-payload
  forecast exactly. Their combined planned RAM stays within the declared cap.
  A 4 GiB budget still rejects before context construction.
- The unchanged-configuration CPU 33-token control preserves all 8,194,560
  logits, expert routes and input IDs byte-for-byte against the previous
  runtime. External model-shard identities are unchanged. The original
  full-prefix and cached references still have 64 and 59 out-of-bound logits;
  all 1,584 ordered expert lists and 33 top-token choices match each reference.
- A separate original CPU 33-token decode changes only the requested expert
  RAM cache from 64 MiB to 1 GiB. All logits, ordered routes and input IDs remain
  byte-identical. Across 47,520 expert-component acquisitions, hits increase
  from zero to 18,442 (38.8%). Logical `pread` bytes fall from 32,414,976,000 to
  19,870,054,400 (38.7%). Both caches stay within their payload capacities and
  drain all leases, active loads and staging reservations. This decode uses
  explicit cache settings without the resource controller; controller admission
  is qualified separately by the initialization checks above. It is one
  diagnostic workload with no matched cold/warm repeats or speedup claim.

These checks establish source-versus-resident planning and empty-context
initialization. They do not establish a physical RSS limit, filled 64K execution,
concurrent serving, general cache-reuse gains or a model speedup. Normal Qwen
loading remains gated. Goldens and fixed tolerances are unchanged.
