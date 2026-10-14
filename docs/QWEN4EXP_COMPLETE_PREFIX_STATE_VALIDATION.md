# Complete recurrent prefix-state primitive

`qwen4exp_build_recurrent_prefix_states` joins convolution, SSM and optional
PLE histories into contiguous F32 `[state_width, tokens]` rows matching the
Qwen cache layout. It performs copying, views and concatenation without new
state arithmetic. Intermediate convolution/SSM columns come from their
existing per-step buffers; final columns come from the operator results.
That producer dependency matters because intermediate journals are written
through leaf inputs as side effects.

Primitive evidence:
`/home/xero/Desktop/test/results/ese-complete-prefix-state-20261011`.
Twenty cases and92accepted cuts cover widths64/128, grouped2/6 and16/48heads,
convolution kernels2/3/4, token counts1/2/3/4/8, absent/small/production-sized
PLE histories and F16 PLE weights. The large case uses128-wide48-head state
and10240PLE channels. Independent convolution/PLE FIFO byte oracles and a
F64 recurrence oracle check every prefix. Poisoned future inputs cannot alter
retained rows. Backend restoration into a padded middle sequence row preserves
its neighbors; scalar continuation from restored rows matches exact component
state and output bytes. Expanding only the complete journal exercises producer
dependencies. CPU and all three GPU native/all-API memory checks pass;14normal
regressions and3ASan/UBSan/leak checks pass.

The first GPU run exposed a real general-convolution checkpoint bug:
`ssm_conv_single_seq_f32` omitted the channel offset in saved-step writes.
Concurrent channels overwrote the same history. Its initial address now includes
`row * (kernel - 1)`. The separately specialized kernel4path already included
that offset. The failing pre-repair binary/source/log and successful repaired
library are preserved. This is a checkpoint correction, not a throughput claim.
The test's initial independent oracle also omitted sigmoid on beta; that failed
test control is retained separately and the oracle now applies sigmoid.

Runtime integration evidence is separate:
`/home/xero/Desktop/test/results/ese-qwen-direct-prefix-20261011`.
Its helper additionally retains the DeltaNet result to prevent ordinary fused
state-copy redirection from leaving the journal's final tail unwritten. The
state-copy-consumer regression passes CPU and all three GPU native/all-API
checks. All28seeded real-model cases at prefixes8/32 pass exact verification,
sampler, sequence state, last logits and future draws. Broader model lifecycle
and all-policy qualification are in progress.
The first implementation is opt-in, one sequence and one verification batch;
sequence banks, microbatch windows, allocation-failure/abort qualification,
MTP hidden-state handling and full resource planning remain required.
