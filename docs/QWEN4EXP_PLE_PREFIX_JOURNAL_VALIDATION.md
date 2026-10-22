# PLE convolution prefix-state primitive

The optional `qwen4exp_build_ple_dilated_conv` overload produces contiguous
F32 histories shaped `[channels, state_length, tokens]`. Entry t holds the
history after input t; `state_length = (kernel_size - 1) * dilation`.
The existing six-argument overload preserves its ABI and arithmetic and
creates no journal. The experimental direct-prefix recovery path uses it behind
`ESE_QWEN4EXP_PREFIX_JOURNAL=1`; the ordinary fallback paths use base replay.

One batched `GET_ROWS` gathers the histories from the existing concatenated
prior/input tensor. A caller-owned contiguous I32 input supplies column indices
`t + h + 1` at `t * state_length + h`. Initialize these fixed indices before
execution/capture and budget their storage with the journal. The primitive does
not add floating-point state arithmetic or per-token host dispatch.

Logical additional storage is `4 * state_length * tokens * (channels + 1)`
bytes for histories and indices. With channels10240, state_length8 and tokens8,
this is2,621,696bytes per layer/window. Allocation rounding, existing history
storage, graph scratch, recurrent journals, sequence banks and other checkpoint
data are separate. Complete startup resource planning remains required.

Evidence is in `/home/xero/Desktop/test/results/ese-ple-prefix-journal-20261011`
and `/home/xero/Desktop/test/results/ese-cuda-graph-binding-20261011`.
The540-case matrix covers channels1/3/32/160/2560/10240, kernels2/3/4,
dilations1/2/4, token counts1/2/3/4/8 and supplied/null prior histories.
It checks every history byte against an independent FIFO oracle, signed-zero
preservation, padded source columns, unchanged ordinary convolution/final-state
bits, poisoned future inputs, and canonical scalar continuation from a retained
prefix. The continuation cut is the halfway prefix (at least one input);
the expanded qualification below now covers every accepted position.

The expanded matrix in
`/home/xero/Desktop/test/results/ese-ple-prefix-all-cuts-20261011` passes1080
F32/F16-weight cases and4968accepted cuts per backend. Cuts run from zero
acceptance through full acceptance, with poisoned suffixes and independent
FIFO histories for canonical scalar continuation. CPU, all three GPU native
runs and all three unsuppressed all-API memory checks pass, zero errors.
Refreshed normal and ASan/UBSan/leak checks also pass. The original540-case
primitive generation and its failed controls remain separately preserved.

CPU and all three GPU native runs pass. With the allocation-binding repair,
all three unsuppressed all-API CUDA memory checks pass540cases each, zero errors.
This is graph-enabled-backend coverage; existing graph compatibility decisions
can choose uncaptured execution for some shapes. A separate graph-disabled
CUDA1memory check passes540cases. Normal and ASan/UBSan/leak CPU checks pass.

Failed controls are retained. GGML rejects a proposed overlapping3Dview because
its packed extent exceeds the source. F32-to-I32 conversion is unsupported in
the attempted index-generation graph. A small-channel CPU scalar/larger-batch
continuation comparison differs by1ULP; scalar continuation now uses independently
constructed FIFO history, while strict journal and same-shape output checks
remain. The original graph-enabled memory check segfaults during `cuGraphLaunch`;
its core backtrace and binding-repair controls are recorded separately.

Direct accepted-prefix recovery still needs complete recurrent/conv state,
rolling token/hash history, sequence-local ownership, checkpoint allocation and
failure handling, QSA/KV truncation, and retained output/hidden-state selection.
Actual MTP, adaptive depth and accepted-token throughput remain unqualified.
