#!/usr/bin/env python3
"""Check actual filled contexts and three-position retrieval on a local server.

This does not start a server or qualify model arithmetic. Keep its server command,
model identity and resource samples alongside the output. Prompt reuse is disabled.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import time
import urllib.parse
import urllib.request

_spec = importlib.util.spec_from_file_location(
    "native_benchmark", Path(__file__).with_name("benchmark-native-inference.py"))
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)

PLACEHOLDER = "ESE_CONTEXT_DOCUMENT_PLACEHOLDER_6b86b273"
FILLER = "\nArchive note: routine maintenance completed; no access codes appear in this note.\n"
CODES = {"ALPHA": "630184", "BETA": "297531", "GAMMA": "845072"}


class ProbeTimeBudgetExceeded(TimeoutError):
    pass


def build_fixture(tokenize, render, length):
    """Build an exact token count without cropping records or the generation prefix."""
    user = ("Read the archive below. Find the ALPHA, BETA and GAMMA access codes. "
            "Reply only with ALPHA=code; BETA=code; GAMMA=code, using the recorded "
            "six-digit numbers.\n<archive>\n" + PLACEHOLDER + "\n</archive>\n"
            "Give all three codes from the archive.")
    rendered = render([{"role": "user", "content": user}])
    if not isinstance(rendered, str) or rendered.count(PLACEHOLDER) != 1:
        raise ValueError("template must preserve exactly one document placeholder")
    before, after = rendered.split(PLACEHOLDER)
    prefix, suffix, filler = tokenize(before), tokenize(after), tokenize(FILLER)
    records = [tokenize(f"\nRECORD: {label} access code is {code}.\n")
               for label, code in CODES.items()]
    if not filler:
        raise ValueError("empty filler tokenization")
    available = length - len(prefix) - len(suffix) - sum(map(len, records))
    if available < 100:
        raise ValueError("requested context leaves insufficient document space")
    gaps = [available // 100, available * 49 // 100, available * 49 // 100]
    gaps.append(available - sum(gaps))
    tokens, positions = list(prefix), {}
    for index, gap in enumerate(gaps):
        tokens.extend((filler * ((gap + len(filler) - 1) // len(filler)))[:gap])
        if index < len(records):
            positions[list(CODES)[index]] = len(tokens)
            tokens.extend(records[index])
    tokens.extend(suffix)
    if len(tokens) != length or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("invalid frozen prompt tokens")
    return dict(tokens=tokens, record_positions=positions, expected=CODES.copy(),
                prefix_tokens=len(prefix), suffix_tokens=len(suffix),
                token_sha256=hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest())


def check_retrieval(content, expected):
    # Each exact six-digit value must be assigned to the right label, rather
    # than merely appearing somewhere in an explanation or another label.
    import re
    for label, code in expected.items():
        matches = re.findall(r"\b" + re.escape(label) + r"\s*[:=]\s*([0-9]{6})\b",
                             content, flags=re.IGNORECASE)
        if not matches or any(value != code for value in matches):
            raise RuntimeError("missing or incorrect archive code: " + label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="existing loopback native HTTP server")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, required=True, help="actual server context per slot")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048])
    parser.add_argument("--predict", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--max-seconds", type=float, default=120,
                        help="hard wall-time budget for the complete probe (POSIX; default: 120)")
    args = parser.parse_args()
    url = urllib.parse.urlsplit(args.url)
    if (url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1", "::1")
            or url.username or url.password or url.path not in ("", "/") or url.query or url.fragment):
        parser.error("use a loopback HTTP server URL without a path or credentials")
    if (min(args.lengths + [args.context, args.predict, args.repeats, args.timeout]) < 1
            or max(args.lengths) + args.predict > args.context):
        parser.error("positive counts and full generation headroom are required")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("wall-time budget must be positive and finite")
    if not hasattr(signal, "setitimer"):
        parser.error("the hard wall-time budget requires POSIX interval timers")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", scope="filled-context retrieval and repeatability only",
                  url=args.url, context_per_slot=args.context, predict=args.predict,
                  requested_lengths=args.lengths, repeats=args.repeats,
                  max_seconds=args.max_seconds, cache_prompt=False, samples=[])

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def request(path, body=None):
        req = urllib.request.Request(args.url.rstrip("/") + path,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=args.timeout)

    def json_request(path, body=None):
        with request(path, body) as response:
            return json.load(response)

    def deadline_expired(signum, frame):
        raise ProbeTimeBudgetExceeded("context probe wall-time budget exhausted")

    save()
    old_alarm = signal.signal(signal.SIGALRM, deadline_expired)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    try:
        report["properties"] = json_request("/props")
        save()
        for length in args.lengths:
            case = args.output / f"prompt-{length}"
            case.mkdir()
            fixture = build_fixture(
                lambda text: json_request("/tokenize", {"content": text, "add_special": False})["tokens"],
                lambda messages: json_request("/apply-template", {
                    "messages": messages, "chat_template_kwargs": {"enable_thinking": False}})["prompt"],
                length)
            (case / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
            text = json_request("/detokenize", {"tokens": fixture["tokens"]})["content"]
            for label, code in CODES.items():
                if text.count(f"{label} access code is {code}.") != 1:
                    raise RuntimeError("record lost or duplicated in frozen token prompt")
            (case / "prompt.txt").write_text(text)
            signature = None
            for repeat in range(args.repeats):
                body = dict(prompt=fixture["tokens"], n_predict=args.predict, temperature=0,
                            seed=4242, ignore_eos=True, cache_prompt=False, stream=True)
                (case / f"request-{repeat + 1}.json").write_text(json.dumps(body) + "\n")
                tick = time.monotonic()
                with request("/completion", body) as response, (case / f"stream-{repeat + 1}.jsonl").open("wb") as raw:
                    content, final, first_token, first_text = benchmark.consume_stream(response, tick, raw)
                # Preserve even a failing receipt/output before asserting its validity.
                (case / f"response-{repeat + 1}.json").write_text(json.dumps(
                    dict(content=content, response=final), indent=2) + "\n")
                timings = benchmark.validate_completion(final, length, args.predict)
                check_retrieval(content, fixture["expected"])
                digest = hashlib.sha256(content.encode()).hexdigest()
                if signature is not None and signature != digest:
                    raise RuntimeError("repeated request changed generated output")
                signature = digest
                report["samples"].append(dict(prompt_tokens=length, repeat=repeat,
                    wall_seconds=time.monotonic() - tick, first_token_seconds=first_token,
                    first_text_seconds=first_text, timings=timings, output_sha256=digest,
                    retrieval_passed=True, truncated=final["truncated"]))
                save()
                print(f"prompt={length} repeat={repeat + 1} retrieval/counts/replay passed", flush=True)
        report["status"] = "passed"
    except KeyboardInterrupt:
        report.update(status="interrupted", error="operator interrupted the probe")
        raise
    except ProbeTimeBudgetExceeded as error:
        report.update(status="time_budget_exceeded", error=str(error))
        raise
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
