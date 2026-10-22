# Next release — GLM Flash and Project Maya

Deferred from the current release by explicit user request on 2026-10-11.
The current release does not include GLM inference or Maya integration. Local
research is preserved separately and does not establish release support here.

## Inputs already identified

- Read-only model directory:
  `/run/media/xero/a44c0fc9-f166-43ec-a0e0-0ae31590045e/glmflash`.
- Four `GLM-5.3-Flash-Maya-L-IQ3_S` GGUF shards, previously inventoried at
  156,319,213,408 bytes in total. Revalidate their identity before use.
- [Project Maya](https://github.com/mw00/project-maya), previously reviewed
  at `6af7f8d8d3c4524c05f609232f93369c60a5755e`, with an MIT license.
  Recheck the source and license before adopting code.

## Follow-up tasks

- [ ] Pin shard metadata, tensor layouts, quantization, checksums and architecture
  compatibility without changing model files.
- [ ] Pin Maya's source/build and run a repeatable baseline on the same shards
  and hardware, retaining raw logs, settings and resource measurements.
- [ ] Map reusable expert loading/cache, graph and dispatch infrastructure;
  inspect GLM-specific KDA, MLA/DSA, mHC and router arithmetic.
- [ ] Port or improve useful implementations with license attribution,
  reference-logit/state parity, lifecycle recovery and bounded-memory checks.
- [ ] Qualify speculation independently and run matched performance ablations.
- [ ] Publish only capabilities supported by the resulting model-backed evidence.

The current ESE/Qwen release does not depend on these tasks.
