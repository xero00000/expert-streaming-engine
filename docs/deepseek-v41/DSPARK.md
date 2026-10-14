# Native DSpark experiment

The local build now includes `deepseek-v41-speculative`, an experimental greedy
decoder for the released three-stage DSpark model. It supports the native V4.1
target and its tokenizer. It is separate from the generic DFlash implementation
and is not integrated into the chat server.

The implementation captures the mean residual streams entering target layers
37, 38 and 39, projects them into DSpark's three attention windows, and proposes
five tokens with the released MXFP4 experts and Markov head. Only accepted
target features update those windows. Target verification uses bounded
checkpoints that restore the exact accepted cache, compression and Engram-hash
prefix, including across window wraps. This path currently requires `--temp 0`.

## Installed draft package

`/home/xero/Desktop/test/models/deepseek-v41-dspark-native/DeepSeek-V4.1-DSpark-MXFP4.gguf`

The 78-tensor package contains the original MXFP4 experts, losslessly repacked
into GGML blocks, and FP8 dense weights expanded to BF16. The target supplies
the shared token embedding and output head. `package.json` records source and
payload hashes. Reproduction uses `tools/deepseek_v41_dspark_package.py` and the
pinned source ranges in
`/mnt/games_sdb2/ai/ese-models/deepseek-v41/dspark-original`.

The draft places stages 0–1 on CUDA0 (5,368 MiB of weights) and stage 2 on CUDA1
(2,735 MiB). It therefore needs a different memory arrangement from the fastest
ordinary-decoding profile, which uses CUDA0 for target matrices.

## Reproduce the validated greedy experiment

Build with `cmake --build build_cuda_release --target deepseek-v41-speculative`.
Use the normal native-target arguments from `RUN_LOCAL.md`, with these changes:

- Select `build_cuda_release/bin/deepseek-v41-speculative` as the executable.
- Use `-ts 1,0,9`, retain `--override-tensor '^output$=CUDA1'`, and set
  `ESE_V41_EXPERT_DEVICE=CUDA2` to leave room for the draft.
- Remove the grouped `^blk...=CUDA0` override from the normal fast profile;
  the draft needs that card's memory. The recorded comparison used 8 GiB of
  pageable expert RAM, two loading workers and the original vocabulary-head
  path, so omit `--pinned-ram` and `--direct-bf16-head` when reproducing it.
- Use `--expert-vram-cache-mib 2048`, `-ub 8`, `-b 32`, and `--temp 0`.
- Add `--dspark-model` with the package path above, `--dspark-proposals 1`,
  and `--spec-stats /absolute/path/stats.json`.
- Keep `ESE_CUDA_MOE_BATCH_VECTOR=1` and `ESE_CUDA_DENSE_BATCH_VECTOR=1`.

The prompt must already use the official chat encoding. For a complete recorded
invocation, see `speed2-dspark-one-fixed/command.json` under
`/home/xero/Desktop/test/results/deepseek-v41-20260910`.

Omit `--dspark-model` to run the same executable as an ordinary greedy baseline.
The statistics file records every generated token ID, proposal, accepted-prefix
length, and the time spent drafting, verifying and updating draft state.

## Measurements and limits

On one short sky-explanation prompt, one-proposal DSpark accepted 18 of 27
proposals and reproduced all baseline token IDs, including EOS. It generated
44 visible tokens in 35.50 seconds; the corresponding ordinary run took
35.63 seconds. This is effectively a tie, not a demonstrated speedup.

A missing BF16 row-gather CUDA path initially forced the Markov embedding table
through the CPU. Implementing that operation reduced measured draft time from
about 448 ms to 54 ms per cycle. The earlier five-proposal run still spent too
much time verifying rejected tokens and was slower overall.

The confidence head is packaged but is not yet used to select proposal counts.
Sampling at nonzero temperature, concurrent sequences, broad acceptance-rate
calibration and server integration remain unverified. Ordinary decoding remains
the recommended speed path on this machine.
