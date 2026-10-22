# Intermediate-channel expert tiles: first execution study

October 10, 2026. External component pilot; no model execution or storage-format change is installed.

The pilot uses the captured layer-20/expert-10 input and original production IQ3_XXS gate/up plus Q2_0 down weights. It rearranges owned one-expert copies into matching gate rows, up rows and down columns. All original weight bytes reconstruct exactly. Q2_0 uses 64-element, 18-byte blocks; channel widths 64/128/256 respect those boundaries. Both the full-expert and tile files contain exactly 1,715,200 weight bytes.

A persistent I/O worker prefetches the next bounded tile while the GPU computes the current one. The baseline also overlaps down-weight loading with gate/up GPU compute. This compares against an existing-overlap baseline, rather than a serialized read-then-compute sequence. Gate/up use the native fused expert operator; down contributions accumulate on the GPU. The pilot checks actual CUDA operator support and every timed output against its corresponding validated result.

## Correctness and resources

On the 3060 Ti, 2080 SUPER and 3080, gate/up outputs are bit-identical to the full expert. The down projection differs by at most **1.1176e-8 absolute / 8.5012e-8 normalized RMS** because the tile accumulation changes reduction order. This is not a bit-identical expert implementation, and it does not qualify full-model logits/routes or arbitrary inputs. The complete baseline matches the archived native backend output exactly.

Unsuppressed compute-sanitizer memcheck, including all API errors, passes on all three devices with zero errors. Instrumented timings are excluded from performance results.

The full expert owns 2,084,096 tensor-buffer bytes. Widths 64/128/256 own 494,336 / 670,976 / 1,695,232 bytes respectively; width 256 includes a separate 128-channel tail operator. Host read-buffer upper bounds are 343,040 / 686,080 / 1,372,160 bytes, versus 1,715,200 for the full expert. These figures exclude CUDA context, runtime pools and executable graphs; they are not total peak GPU memory.

## Timing

Seven alternating-order full/tiled pairs per condition and device, after warming both kernels and their graphs. Cold advisory applies only to the owned probe copies on NVMe. Process I/O counters confirm **1,716,224 physical bytes for every cold run** and **zero physical bytes for every warm run**; requested weight bytes stay constant. Clocks and background activity were uncontrolled.

Full load-plus-compute time change, positive means slower:

| GPU | Channels | Warm | Cold |
|---|---:|---:|---:|
| 3060 Ti | 64 | +88.5% | +22.7% |
| 3060 Ti | 128 | +38.7% | +2.7% |
| 3060 Ti | 256 | +20.9% | +16.2% |
| 2080 SUPER | 64 | +26.3% | −5.5% |
| 2080 SUPER | 128 | +12.0% | −1.4% |
| 2080 SUPER | 256 | +5.7% | +4.9% |
| 3080 | 64 | +140.2% | +23.7% |
| 3080 | 128 | +74.6% | +11.6% |
| 3080 | 256 | +60.7% | +14.4% |

Tiles begin useful work earlier. For example, warm 3060 Ti runs finish the first 64-channel hidden slice around 150 µs, versus about 627 µs for the full 640-channel hidden state. Those are different amounts of completed work; the first-slice result is not a throughput multiplier. Overall expert completion is generally slower. Small cold gains on Turing do not establish a general win.

The smaller working set is interesting, but this straightforward native-operator pipeline is not justified as a speed optimization. A deeper fused implementation would need to recover launch/reduction efficiency and pass full-model arithmetic/lifecycle gates before integration.

Evidence: `/home/xero/Desktop/test/results/ese-channel-tiles-20261010` contains the layout/reconstruction manifest, actual quantized probe copies, persistent-worker pilot source, validated output/timing logs, exact physical-read checks, per-device sanitizer logs and `tile-analysis.json`. Initial first-use/capture timings are retained as explicitly unqualified diagnostics; they are excluded from this table.
