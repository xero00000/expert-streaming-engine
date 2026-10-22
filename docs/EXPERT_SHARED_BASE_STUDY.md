# Exact shared-base expert blocks: first compression study

October 10, 2026. Offline experiment; the tested schemes are rejected for this artifact. Model bytes and inference storage are unchanged.

The study reads matching 32-row windows from eight experts at a time, using both adjacent and dispersed expert IDs. It covers 96 groups across 16 representative projection/quantization selections in the production Qwen GGUF, including gate/up/down and later layers: 21,934,080 sampled original bytes. First, middle and final row windows are included.

Each expert block is either independently Zstandard-compressed or represented by an exact shared base plus XOR/modular-byte-subtraction residuals. The base is stored once per group, and frame/group metadata is charged. Both the first expert and an aligned-byte medoid are considered as bases. Every reconstructed block is checked against its original bytes. Cold random access charges fetching and reconstructing the base for each requested residual.

At compression levels 1 and 3, fixed-first-base residual schemes occupy 21,941,202 bytes, versus approximately 21,842,434 bytes for independent frames. They are about **0.45% larger**, including metadata. The median cold random-access encoded-byte amplification is **1.875×**. Exploratory reconstruction is about 2.6× slower than independently decompressing a block; these Python/NumPy timings include verification and are not an optimized decoder benchmark.

The simple schemes show no transfer-saving case here. Similar-looking experts have not produced useful exact redundancy in the tested quantized representation. This rejects these schemes on these sampled blocks, rather than proving that every possible shared-base method fails. Do not invest in runtime integration without materially better offline compression and access-cost evidence.

Evidence and reproduction: `/home/xero/Desktop/test/results/ese-model-plan-20261010/compression-study.py`, `compression-study.json`, `compression-block-results.json`, source inventory and tensor index. The report records exact payload hashes, family/row/expert selections, compressed sizes, extra reconstruction scratch and random-access amplification. No source model is copied into the repository.
