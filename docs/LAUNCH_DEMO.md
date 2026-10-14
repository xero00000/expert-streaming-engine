# Reproducible launch demo

**Status: qualification pending. No low-RAM or filled-64K headline is established.**

The launch aims to show a supported large sparse model whose weights exceed
both available RAM and VRAM. The next model download is user-managed. Historical
GPT-OSS 120B results are a comparison/control, not a new flagship result.

## What the published record must contain

| Field | Required evidence |
| --- | --- |
| Exact model | Publisher, pinned repository revision, complete filenames/shards and SHA-256 hashes |
| Quantization | GGUF tensor-type breakdown as well as the publisher’s quant label; no “F16” claim from a filename alone |
| Hardware | CPU, installed/available RAM, selected GPU(s), per-card VRAM, driver, OS and SSD identity |
| Runtime | ESE source generation, package/native hashes, launch command and relevant environment |
| Context | At least 65,536 actual prompt tokens; capacity leaves room for generated output; no silent truncation/shift |
| Memory | Peak process RSS, GPU process peaks, KV/cache settings, memory-limit and swap configuration; page-cache effects disclosed |
| Speed | Startup, prompt processing, first response and sustained output; counts, duration, repeated requests and cold/warm condition |
| Useful context | Retrieval at the beginning, middle and end, with prompts and responses preserved |
| Failure record | OOMs, timeouts, budget rejections and failed retrievals; successful settings do not erase failed trials |
| Reproduction | Download instructions, exact commands, request data and raw machine-readable results |

All model inference for this launch runs from SSD/NVMe. Hard-drive storage may
hold archives and original files; it is not the benchmark source.

## Claim policy

A model that loads is not a sustained-speed result. A context allocation is not
a filled-context result. An expert RAM-cache budget is not the whole memory
footprint. The launch headline will use measured numbers and name its hardware
rather than promising fast execution on every PC.

Production-disabled architectures stay clearly experimental. Their private
probe results do not turn them into supported release demonstrations.
