# Native filled-context qualification

Task 5 checks actual prompt processing, retrieval and session behavior after the
user accepted the current Flash-Next decode speed. A configured 1M context is
an allocation setting; it does not prove successful use of that many tokens.
The selected model remains behind the separate experimental build/run gate.

`scripts/qualify-native-context.py` runs against an existing local native HTTP
server. Start that server with the recorded candidate runtime, exact model,
cache policy and GPU placement. Retain its command, environment, source/library
hashes, server log and sampled process/GPU memory alongside the probe output.
The probe does not launch, reconfigure or terminate that server.

Example for a server with 1,048,576 context cells per slot:

```sh
python3 scripts/qualify-native-context.py \
  --url http://127.0.0.1:8080 \
  --context 1048576 \
  --lengths 2048 8192 65536 \
  --predict 256 --repeats 2 \
  --output /absolute/path/to/new-evidence-directory
```

The probe uses the server's chat template and tokenizer, freezes exact token-ID
prompts, and places labelled six-digit records near the start, middle and end.
Filler alone is cropped to reach the requested length. The generation prefix
and complete records are retained and verified by detokenization. The expected
answers appear only in those records. This is a synthetic retrieval workload,
not a general model-quality evaluation.

Every completion disables prompt reuse, uses greedy sampling with a fixed seed,
and requests a fixed output count with EOS ignored. The final receipt must
confirm the complete prompt and output counts, finite consistent timings and
no truncation. Each answer must assign every code to the correct label. Repeats
must preserve the generated text exactly. Raw streams and final responses are
saved before validation; a failed check marks the report failed and stops the
run. Existing evidence directories are never replaced.

Begin with the 2K pilot before the 8K/65,536-token runs. Long prefill can take
substantially longer than decode; the default per-request socket timeout is two
hours. Keep the server's own resource/cancellation limits active. Record failed
or aborted trials separately. Two successful retrieval responses do not prove
arbitrary concurrency, full-history numerical parity, memory bounds or a 1M
filled-context claim. Interleaved-session and resource acceptance need their
own evidence. Do not open public model admission based on this probe alone.
