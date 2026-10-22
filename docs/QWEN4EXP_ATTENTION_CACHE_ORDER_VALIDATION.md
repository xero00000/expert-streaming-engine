# Ordered cache rows: component qualification

October 11, 2026. This component is qualified; model-graph integration and
two-sequence prefix-bank recovery remain incomplete.

The captured first divergence in the bank continuation test is layer-3 flash
attention. Query bytes and the twelve logical, visible F16 key/value rows are
identical, but physical positions differ. The original attention operator
changes 1,572 of 6,144 output values, with maximum absolute difference
0.00108456611633. The same difference reproduces on all three GPUs.

`qwen4exp_build_cache_rows` gathers explicitly supplied row IDs from a cache
with shape `[channels, physical rows, heads]`. It preserves F16/F32 storage
bits using native same-type GET_ROWS, without converting F16 through an F32
intermediate or copying the complete source cache. IDs are I32
`[ordered rows, heads]`; callers must supply valid physical indices, sequence
order and masks for repeated padding rows. The helper does not yet select
sequence rows or replace the experimental model's attention graph.

The regression uncovered an existing CUDA same-type GET_ROWS defect. Its
dim1 path copied a scalar-channel count as uint32 chunks for even-width F16
rows, overreading/overwriting rows. It also measured I32 index strides in copy
chunk units, which is wrong when copying uint16 chunks. The repair copies
`row_bytes / chunk_bytes`, uses sizeof(int32_t) for index strides, and chooses
uint16 chunks when addresses or row/head strides cannot support uint32
alignment. The dim0 path is unchanged and outside this qualification.

The identical final test executable fails with the frozen old GGML library:
stored row bits change in a 63-channel, two-head F16 case. A separate earlier
64-channel old-library memcheck reports 34 errors, including invalid global
writes; both failed controls are retained.

Final qualification uses immutable v4 libraries and a recorded resolution of
their patched runtime search paths:

- 54 stored-bit gather cases on CPU and each of the three GPUs: F16/F32,
  widths 63/64/256, one/two/four heads, 1/17/256 output rows, reversed and
  repeated indices, padded token/head strides and nonzero view offsets.
  The 17-row cases include a one-scalar offset, testing unaligned F16 starts.
- The captured attention pair on every GPU: all ordered K/V bytes and all
  6,144 attention output bits agree. Masked padding repeats the first valid
  logical row. The physical-placement negative control still fails exact
  agreement on every device.
- Independent dense F64 oracle on the original visible rows: finite output
  and the existing DEFAULT-attention tolerance `0.002 * (1 + abs(expected))`
  pass. Maximum absolute oracle error is 0.00170364810781. Ordered attention
  changes the reduction order; it is not bit-identical to either original
  physically positioned output and does not improve every oracle error.
- Unsuppressed all-API CUDA memcheck passes on all three GPUs with zero
  errors. Nine normal CTests and the CPU gather test under strict
  ASan/UBSan/leak detection pass.

The source and binary snapshots, logs, failures and machine-checked
`qualification-audit.json` are under
`/home/xero/Desktop/test/results/ese-attention-cache-order-20261011`.
The first attempted zero-stride ID view exceeded GGML's view-construction
size check. That failure remains separate from the explicit per-head IDs.
The initial probe also resolved build-tree libraries through DT_RPATH;
accepted v2/v3/v4 runs explicitly resolve their frozen libraries instead.

For one ordered F16 K/V pair at width 256, two KV heads and 256 rows, the
helper outputs own 524,288 tensor bytes and the shared ID input owns 2,048
bytes. At 64K rows those amounts become 128 MiB and 512 KiB. These are
component tensor sizes, excluding allocation alignment, attention workspace,
source-cache storage and other live graph tensors. No complete peak-memory
or throughput result is claimed.

Next, integrate sequence-owned ordering and causal masks into the Qwen graph,
including sequence-aware QSA pooling, positions and tail selection. The
ordered key length must depend on that sequence's retained keys rather than
another sequence's physical cache extent, and allocation/reuse must respect
the resource budget. Then rerun the original target0/accepted4 bank failure,
all bank cuts, ordinary teacher controls, cancellation and memory checks.
Full upstream parity, hybrid expert arithmetic, long-context and release
qualification remain open.
