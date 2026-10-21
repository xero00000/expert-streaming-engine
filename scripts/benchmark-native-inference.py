#!/usr/bin/env python3
"""Record exploratory native HTTP latency, throughput, repeatability and memory evidence.

No performance acceptance threshold is inferred. Fresh processes reuse the OS
file cache; model hashing precedes timing. Prompt KV reuse is disabled.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.request


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def file_residency(path):
    """Read Linux page residency without touching file contents or evicting pages."""
    try:
        page = os.sysconf('SC_PAGE_SIZE')
        size = path.stat().st_size
        with path.open('rb') as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_COPY) as mapping:
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            pages = (size + page - 1)//page
            vector = (ctypes.c_ubyte * pages)()
            libc = ctypes.CDLL(None, use_errno=True)
            libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            if libc.mincore(address, size, vector):
                raise OSError(ctypes.get_errno(), 'mincore')
            resident = sum(v & 1 for v in vector)
        return dict(pages=pages, resident_pages=resident, resident_fraction=resident/pages)
    except (OSError, ValueError, AttributeError) as error:
        return dict(unavailable=str(error))


def process_memory(pid):
    result = {}
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmHWM', 'RssAnon', 'RssFile', 'VmSwap'):
                result[key+'_bytes'] = int(value.split()[0])*1024
        result['io'] = {k: int(v) for k, v in (line.split(':') for line in Path(f'/proc/{pid}/io').read_text().splitlines())}
    except (OSError, ValueError):
        pass
    return result


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu',
        '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=10)
    if result.returncode:
        return dict(error=result.stderr.strip())
    rows = []
    for line in result.stdout.splitlines():
        values = [v.strip() for v in line.split(',')]
        rows.append(dict(zip(('uuid', 'total_mib', 'used_mib', 'utilization_percent', 'power_watts', 'temperature_c'), values)))
    return rows


def validate_completion(final, prompt_tokens, predicted):
    if not final.get('stop') or final.get('truncated'):
        raise RuntimeError('incomplete or truncated completion')
    timing = final.get('timings', {})
    if (final.get('tokens_predicted') != predicted or final.get('tokens_evaluated') != prompt_tokens or
            timing.get('predicted_n') != predicted or timing.get('prompt_n') != prompt_tokens):
        raise RuntimeError('actual prompt/generation counts differ from the frozen workload')
    for key in ('prompt_ms', 'predicted_ms', 'prompt_per_second', 'predicted_per_second'):
        if not isinstance(timing.get(key), (int, float)) or not math.isfinite(timing[key]) or timing[key] <= 0:
            raise RuntimeError('invalid server timing: '+key)
    return timing


def consume_stream(response, started, raw):
    chunks, first_token, first_text, final = [], None, None, None
    for line in response:
        raw.write(line)
        if not line.startswith(b'data:'):
            continue
        body = line[5:].strip()
        if not body or body == b'[DONE]':
            continue
        value = json.loads(body)
        if 'error' in value:
            raise RuntimeError(str(value['error']))
        now = time.monotonic()-started
        if not value.get('stop'):
            if first_token is None and (value.get('tokens_predicted', 0) > 0 or value.get('content')):
                first_token = now
            if value.get('content'):
                if first_text is None:
                    first_text = now
                chunks.append(value['content'])
        else:
            final = value
    if final is None or first_token is None:
        raise RuntimeError('stream ended without a token and final receipt')
    return ''.join(chunks), final, first_token, first_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prompt-json', type=Path, required=True, help='JSON object with frozen token IDs under tokens')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', choices=('resident', 'cache', 'stream'), required=True)
    parser.add_argument('--gpus', required=True, help='comma-separated physical GPU UUIDs')
    parser.add_argument('--tensor-split', default='1,1,1.2')
    parser.add_argument('--starts', type=int, default=3)
    parser.add_argument('--requests', type=int, default=3)
    parser.add_argument('--predict', type=int, default=128)
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--max-ram', default='32GiB')
    parser.add_argument('--reserve-vram', default='1GiB')
    parser.add_argument('--ram-cache-mib', type=int, default=1024)
    parser.add_argument('--vram-cache-mib', type=int, default=1024)
    parser.add_argument('--staging-mib', type=int, default=64)
    parser.add_argument('--prefill-staging-mib', default='0', help='MiB per GPU, or auto for native sizing')
    parser.add_argument('--server-arg', action='append', default=[])
    args = parser.parse_args()
    if min(args.starts, args.requests, args.predict, args.context, args.threads) < 1:
        parser.error('counts must be positive')
    server, model = args.server.resolve(strict=True), args.model.resolve(strict=True)
    prompt = json.loads(args.prompt_json.read_text())['tokens']
    if not prompt or any(type(token) is not int or token < 0 for token in prompt) or len(prompt)+args.predict > args.context:
        parser.error('invalid prompt tokens or context headroom')
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()
    (out/'prompt.json').write_text(args.prompt_json.read_text())
    report = dict(status='running', scope='measurement only; matched Strata comparison and numerical acceptance required',
        policy=args.policy, source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        server_sha256=sha256(server), model_sha256=sha256(model), model_bytes=model.stat().st_size,
        prompt_tokens=len(prompt), prompt_sha256=hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest(),
        starts=[], samples=[], monitor_interval_seconds=1,
        cache_control='hashing warms model file cache; fresh process per start; no cache eviction; cache_prompt=false')
    report['runner_sha256'] = sha256(Path(__file__))
    version = subprocess.run([str(server), '--version'], text=True, capture_output=True)
    report['server_version'] = version.stdout + version.stderr
    identity = (model.stat().st_size, model.stat().st_mtime_ns)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
    environment.pop('LD_PRELOAD', None)
    report['environment'] = {k:v for k,v in environment.items() if k.startswith(('CUDA_', 'LD_', 'ESE_', 'GGML_'))}
    signature = None
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for iteration in range(args.starts):
            if (model.stat().st_size, model.stat().st_mtime_ns) != identity:
                raise RuntimeError('model file changed')
            run = out/f'start-{iteration+1}';run.mkdir()
            with socket.socket() as listener:
                listener.bind(('127.0.0.1',0)); port = listener.getsockname()[1]
            command = [str(server), '-m', str(model), '--memory-policy', args.policy, '--max-ram', args.max_ram,
                '--reserve-vram', args.reserve_vram, '--max-context', str(args.context), '--min-kv-quality', 'f16',
                '-c', str(args.context), '-b', '512', '-ub', '128', '-np', '1', '-t', str(args.threads), '-tb', str(args.threads),
                '-ngl', '99', '-sm', 'layer', '-ts', args.tensor_split, '--cache-ram', '64', '--no-warmup',
                '--no-fused-mul-multiadd', '--host', '127.0.0.1', '--port', str(port)]
            if args.policy != 'resident':
                command += ['--cpu-moe', '--expert-ram-cache-mib', str(args.ram_cache_mib),
                    '--expert-ram-staging-mib', str(args.staging_mib), '--expert-vram-cache-mib', str(args.vram_cache_mib),
                    '--expert-load-workers', '2', '--expert-storage-backend', 'mmap' if args.policy=='cache' else 'pread',
                    '--expert-sidecar-only']
                if args.prefill_staging_mib != 'auto':
                    command += ['--expert-prefill-staging-mib', args.prefill_staging_mib]
            command += args.server_arg
            entry = dict(iteration=iteration, argv=command, file_cache_before=file_residency(model), ambient_gpu_before=gpu_snapshot())
            report['starts'].append(entry);save()
            process = None; done = threading.Event(); monitor = None
            def request(path, payload=None, name=None):
                req = urllib.request.Request(f'http://127.0.0.1:{port}'+path,
                    data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=600) as response: data=json.load(response)
                if name: (run/(name+'.json')).write_text(json.dumps(dict(path=path,payload=payload,response=data),indent=2)+'\n')
                return data
            def watch():
                with (run/'memory-samples.jsonl').open('w') as file:
                    while not done.is_set():
                        row=dict(seconds=time.monotonic()-started,process=process_memory(process.pid))
                        try:row['gpus']=gpu_snapshot()
                        except subprocess.TimeoutExpired:row['gpu_error']='sampling timeout'
                        file.write(json.dumps(row)+'\n');file.flush()
                        done.wait(1)
            try:
                with (run/'server.log').open('w') as log:
                    started=time.monotonic();process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
                    monitor=threading.Thread(target=watch,daemon=True);monitor.start()
                    while True:
                        if process.poll() is not None:raise RuntimeError(f'server startup exit {process.returncode}')
                        try:request('/health');break
                        except OSError:
                            if time.monotonic()-started>300:raise RuntimeError('startup timeout')
                            time.sleep(.1)
                    entry['startup_to_ready_seconds']=time.monotonic()-started
                    request('/v1/ese/resources',name='resources-before')
                    for trial in range(args.requests):
                        payload=dict(prompt=prompt,n_predict=args.predict,n_probs=0,temperature=0,seed=4242,
                            ignore_eos=True,cache_prompt=False,stream=True)
                        (run/f'request-{trial+1}.json').write_text(json.dumps(payload)+'\n')
                        before=process_memory(process.pid);tick=time.monotonic()
                        req=urllib.request.Request(f'http://127.0.0.1:{port}/completion',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                        with urllib.request.urlopen(req,timeout=600) as response,(run/f'response-{trial+1}.sse').open('wb') as raw:
                            content,final,ttft,first_text=consume_stream(response,tick,raw)
                        wall=time.monotonic()-tick
                        timing=validate_completion(final,len(prompt),args.predict)
                        (run/f'final-{trial+1}.json').write_text(json.dumps(dict(content=content,response=final),indent=2)+'\n')
                        value=hashlib.sha256(content.encode()).hexdigest()
                        if signature is None:signature=value
                        row=dict(start=iteration,request=trial,wall_seconds=wall,first_token_seconds=ttft,
                            first_visible_text_seconds=first_text,timings=timing,output_sha256=value,
                            exact_replay=value==signature,process_before=before,process_after=process_memory(process.pid))
                        report['samples'].append(row);save()
                        if not row['exact_replay']:raise RuntimeError('greedy cold-request output changed')
                        request('/v1/ese/resources',name=f'resources-after-{trial+1}')
                        print(json.dumps(dict(policy=args.policy,start=iteration,request=trial,ttft=ttft,decode_tps=timing['predicted_per_second'])),flush=True)
            finally:
                done.set()
                if monitor:monitor.join(timeout=15)
                if process is not None:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        try:process.wait(timeout=30)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
                    entry['server_returncode']=process.returncode
                save()
            if entry.get('server_returncode') != 0:raise RuntimeError('server did not shut down cleanly')
        report['summary'] = dict(
            startup_to_ready_seconds=[s['startup_to_ready_seconds'] for s in report['starts']],
            decode_tps_median=statistics.median(s['timings']['predicted_per_second'] for s in report['samples']),
            prefill_tps_median=statistics.median(s['timings']['prompt_per_second'] for s in report['samples']),
            first_token_seconds_median=statistics.median(s['first_token_seconds'] for s in report['samples']),
            first_request_decode_tps=[s['timings']['predicted_per_second'] for s in report['samples'] if s['request']==0],
            repeated_request_decode_tps=[s['timings']['predicted_per_second'] for s in report['samples'] if s['request']>0])
        report['status']='measurements_complete'
    except Exception as error:
        report.update(status='failed',error=str(error));raise
    finally:save()
    return 0


if __name__=='__main__':raise SystemExit(main())
