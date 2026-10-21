#!/usr/bin/env python3
"""Measure a local Strata qwen35 engine without changing its service or config.

Uses the same frozen prompt format as benchmark-native-inference.py. Engine
protocol TTFT excludes HTTP; compare engine decode counters separately from it.
No performance or numerical acceptance is inferred from these measurements.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import threading
import time

spec = importlib.util.spec_from_file_location('native_benchmark', Path(__file__).with_name('benchmark-native-inference.py'))
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def parse_receipt(line, tokens, prompt_count, predicted):
    fields = line.split()
    if len(fields) != 16 or fields[0] != 'DONE':
        raise RuntimeError('unexpected Strata DONE format')
    count, prompt_n, reused, fresh = (int(fields[i]) for i in (1, 2, 8, 14))
    pp_ms, tg_ms = float(fields[3]), float(fields[4])
    if count != predicted or len(tokens) != predicted or prompt_n != prompt_count or reused != 0 or fresh != prompt_count or fields[5] != 'length':
        raise RuntimeError('incomplete generation, reused prompt, or workload count mismatch')
    import math
    if not all(math.isfinite(v) and v > 0 for v in (pp_ms, tg_ms)):
        raise RuntimeError('invalid Strata timing')
    return dict(prompt_n=prompt_n, predicted_n=count, prompt_ms=pp_ms, predicted_ms=tg_ms,
                prompt_per_second=prompt_n*1000/pp_ms, predicted_per_second=count*1000/tg_ms,
                prompt_reused=reused, drafts_accepted=int(fields[6]), drafts_offered=int(fields[7]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prompt-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--starts', type=int, default=3)
    parser.add_argument('--requests', type=int, default=3)
    parser.add_argument('--predict', type=int, default=128)
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--expert-vram', default='4', help='GiB; Strata controls its remaining allocations')
    args = parser.parse_args()
    prompt = json.loads(args.prompt_json.read_text())['tokens']
    if min(args.starts, args.requests, args.predict, args.threads) < 1 or not prompt or len(prompt)+args.predict > args.context:
        parser.error('invalid counts/context')
    if any(type(t) is not int or t < 0 for t in prompt): parser.error('invalid token IDs')
    engine, model = args.engine.resolve(strict=True), args.model.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()
    (out/'prompt.json').write_text(args.prompt_json.read_text())
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    for key in ('LD_PRELOAD', 'LD_LIBRARY_PATH'): environment.pop(key, None)
    report = dict(status='running', scope='measurement only; no numerical/performance acceptance',
        engine=str(engine), engine_sha256=native.sha256(engine), model=str(model),
        model_sha256=native.sha256(model), runner_sha256=native.sha256(Path(__file__)),
        engine_version=subprocess.run([str(engine),'--version'],text=True,capture_output=True,env=environment).stdout,
        prompt_tokens=len(prompt), prompt_sha256=native.sha256(out/'prompt.json'),
        cache_control='model hashing before timing; OS cache retained; DONE must report zero reused tokens',
        environment={k:v for k,v in environment.items() if k.startswith(('CUDA_','LD_','STRATA_'))},
        starts=[],samples=[])
    identity=(model.stat().st_size, model.stat().st_mtime_ns)
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for iteration in range(args.starts):
            if (model.stat().st_size, model.stat().st_mtime_ns) != identity: raise RuntimeError('model changed')
            run=out/f'start-{iteration+1}'; run.mkdir()
            command=[str(engine),'--serve','--model',str(model),'--max-context',str(args.context),
                '--chunk','512','--mtp','0','--cpu-threads',str(args.threads),
                '--expert-vram',args.expert_vram,'--no-warmup']
            entry=dict(argv=command, file_cache_before=native.file_residency(model), ambient_gpu_before=native.gpu_snapshot())
            report['starts'].append(entry);save()
            process=None;done=threading.Event();monitor=None;reader=None;q=queue.Queue()
            with (run/'stderr.log').open('w') as err,(run/'stdout.log').open('w') as raw:
                def pump():
                    for line in process.stdout: raw.write(line);raw.flush();q.put(line.rstrip())
                    q.put(None)
                def line():
                    try:value=q.get(timeout=300)
                    except queue.Empty:raise RuntimeError('Strata protocol timeout')
                    if value is None:raise RuntimeError('Strata ended unexpectedly')
                    if value.startswith('ERR'):raise RuntimeError(value)
                    return value
                def watch():
                    with (run/'memory-samples.jsonl').open('w') as log:
                        while not done.is_set():
                            value=dict(seconds=time.monotonic()-started,process=native.process_memory(process.pid))
                            try:value['gpus']=native.gpu_snapshot()
                            except subprocess.TimeoutExpired:value['gpu_error']='timeout'
                            log.write(json.dumps(value)+'\n');log.flush();done.wait(1)
                try:
                    started=time.monotonic()
                    process=subprocess.Popen(command,env=environment,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
                    reader=threading.Thread(target=pump,daemon=True);reader.start()
                    monitor=threading.Thread(target=watch,daemon=True);monitor.start()
                    while not line().startswith('READY'):pass
                    entry['startup_to_ready_seconds']=time.monotonic()-started
                    for trial in range(args.requests):
                        payload='GEN '+str(args.predict)+' temperature=0 seed=4242 '+','.join(map(str,prompt))
                        (run/f'request-{trial+1}.txt').write_text(payload+'\n')
                        before=native.process_memory(process.pid);tokens=[];ttft=None;tick=time.monotonic()
                        process.stdin.write(payload+'\n');process.stdin.flush()
                        while True:
                            value=line()
                            if value.startswith('T '):
                                if ttft is None:ttft=time.monotonic()-tick
                                tokens.append(int(value.split()[1]))
                            if value.startswith('DONE '):break
                        wall=time.monotonic()-tick
                        timing=parse_receipt(value,tokens,len(prompt),args.predict)
                        row=dict(start=iteration,request=trial,tokens=tokens,done=value,wall_seconds=wall,
                            first_token_seconds=ttft,timings=timing,process_before=before,process_after=native.process_memory(process.pid))
                        report['samples'].append(row);save()
                        print(json.dumps(dict(start=iteration,request=trial,ttft=ttft,decode_tps=timing['predicted_per_second'])),flush=True)
                finally:
                    done.set()
                    if monitor:monitor.join(timeout=15)
                    if process:
                        try:
                            process.stdin.close();process.wait(timeout=30)
                        except (OSError,subprocess.TimeoutExpired):process.kill();process.wait()
                        if reader:reader.join(timeout=5)
                        entry['returncode']=process.returncode
                    save()
            if entry.get('returncode')!=0:raise RuntimeError('Strata shutdown failed')
        report['summary']=dict(
            startup_to_ready_seconds=[s['startup_to_ready_seconds'] for s in report['starts']],
            decode_tps_median=statistics.median(s['timings']['predicted_per_second'] for s in report['samples']),
            prefill_tps_median=statistics.median(s['timings']['prompt_per_second'] for s in report['samples']),
            first_token_seconds_median=statistics.median(s['first_token_seconds'] for s in report['samples']),
            unique_output_sequences=len({tuple(s['tokens']) for s in report['samples']}))
        report['status']='measurements_complete'
    except Exception as error:report.update(status='failed',error=str(error));raise
    finally:save()


if __name__=='__main__':main()
