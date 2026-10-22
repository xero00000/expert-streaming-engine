#!/usr/bin/env python3
"""Check real HTTP disconnect cleanup and exact cold retry on a small MoE fixture.

Cache sizes target the retained TinyMoE fixture. This is neither an actual OOM
trial nor a large-model, Qwen, multimodal or performance qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('cpu-resident', 'cpu-stream', 'gpu-cache'), required=True)
    parser.add_argument('--gpu-count', type=int, default=1)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--server-arg', action='append', default=[])
    parser.add_argument('--prompt-token-ids', help='comma-separated token IDs for synthetic tokenizers')
    args = parser.parse_args()
    if args.runs < 1 or args.gpu_count < 1:
        parser.error('runs and GPU count must be positive')
    server, model = args.server.resolve(strict=True), args.model.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    command = [str(server), '-m', str(model), '-c', '1024', '-np', '1', '-b', '128',
               '-ub', '1', '-t', '4', '-tb', '4', '--host', '127.0.0.1', '--port', str(port),
               '--no-warmup', '-ngl', '99' if args.mode == 'gpu-cache' else '0']
    if args.mode != 'cpu-resident':
        command += ['--expert-ram-cache-mib', '8', '--expert-ram-staging-mib', '2',
                    '--expert-sidecar-only', '--defer-experts', '--expert-storage-backend', 'pread',
                    '--expert-load-workers', '2']
    if args.mode == 'cpu-stream':
        command += ['--memory-policy', 'stream', '--max-ram', '4GiB',
                    '--max-context', '1024', '--min-kv-quality', 'f16']
    if args.mode == 'gpu-cache':
        command += ['-ot', '.ffn_.*_exps.=CPU', '--expert-vram-cache-mib', '6',
                    '--expert-vram-reserve-mib', '256', '--expert-cache-min-observations', '2',
                    '-sm', 'layer', '-ts', ','.join('1' for _ in range(args.gpu_count))]
    command += args.server_arg
    environment = os.environ.copy()
    if args.mode != 'gpu-cache':
        environment['CUDA_VISIBLE_DEVICES'] = ''
    report = {'status': 'running', 'argv': command, 'mode': args.mode,
              'model_sha256': sha256(model), 'server_sha256': sha256(server),
              'environment': {k: v for k, v in environment.items()
                              if k.startswith(('CUDA_', 'LD_', 'ESE_', 'GGML_'))}, 'runs': []}
    report_path = out / 'report.json'

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    request_number = 0

    def request(path: str, payload: dict | None = None) -> dict:
        nonlocal request_number
        request_number += 1
        body = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=body,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response:
            value = json.load(response)
        (out / f'{request_number:04d}-response.json').write_text(json.dumps(
            {'path': path, 'payload': payload, 'response': value}, indent=2) + '\n')
        return value

    def resources() -> dict:
        return request('/v1/ese/resources')

    def wait_for(predicate, seconds: float) -> dict:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            snapshot = resources()
            if predicate(snapshot):
                return snapshot
            time.sleep(.02)
        raise RuntimeError('resource state did not reach the required condition')

    def require_drained(snapshot: dict) -> None:
        if not snapshot['safe_point'] or snapshot['slots']['processing'] or snapshot['slots']['deferred']:
            raise RuntimeError('cancelled slot or deferred task remains active')
        ram = snapshot['expert_ram']
        if ram['active_leases'] or ram['resident_bytes'] > ram['capacity_bytes']:
            raise RuntimeError('RAM lease ownership or capacity failed')
        transient = snapshot.get('transient', {})
        if any(transient.get(key, 0) for key in ('active_leases', 'pins', 'pending_restores', 'reconfiguration_open')):
            raise RuntimeError('transient ownership did not drain')
        for device in snapshot['devices']:
            cache = device['expert_cache']
            if cache['capacity_bytes'] and max(cache['allocated_bytes'], cache['resident_bytes']) > cache['capacity_bytes']:
                raise RuntimeError('GPU cache exceeded its bound')

    payload = dict(prompt='The capital of France is', n_predict=16, n_probs=10,
                   temperature=0, seed=4242, ignore_eos=True, cache_prompt=False)

    if args.prompt_token_ids:
        payload['prompt'] = [int(value) for value in args.prompt_token_ids.split(',')]

    def signature(response: dict) -> dict:
        return {key: response[key] for key in ('content', 'completion_probabilities')}

    save()
    process = None
    try:
        with (out / 'server.log').open('w') as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            report['pid'] = process.pid
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError('server exited before becoming ready')
                try:
                    request('/health')
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError('server startup timed out')
                    time.sleep(.1)
            teacher = signature(request('/completion', payload))
            control = signature(request('/completion', payload))
            if control != teacher:
                raise RuntimeError('repeated cold control changed before cancellation')
            require_drained(resources())
            for trial in range(args.runs):
                long_payload = dict(payload, n_predict=512, n_probs=0)
                body = json.dumps(long_payload).encode()
                header = (f'POST /completion HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n'
                          f'Content-Type: application/json\r\nContent-Length: {len(body)}\r\n'
                          'Connection: close\r\n\r\n').encode()
                connection = socket.create_connection(('127.0.0.1', port), timeout=10)
                try:
                    connection.sendall(header + body)
                    busy = wait_for(lambda value: not value['safe_point'], 5)
                    started = time.monotonic()
                    connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
                    connection.shutdown(socket.SHUT_RDWR)
                finally:
                    connection.close()
                drained = wait_for(lambda value: value['safe_point'], 30)
                elapsed = time.monotonic() - started
                require_drained(drained)
                retry = signature(request('/completion', payload))
                if retry != teacher:
                    raise RuntimeError('cold retry content/probabilities changed after disconnect')
                require_drained(resources())
                report['runs'].append({'trial': trial, 'cleanup_seconds': elapsed, 'busy': busy,
                                       'drained': drained, 'exact_retry': True,
                                       'cancel_payload': long_payload})
                save()
    except BaseException as error:
        report.update(status='fail', error=str(error))
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            report['server_exit'] = process.returncode
        save()
    log_text = (out / 'server.log').read_text()
    cancellations = log_text.lower().count('cancel task')
    if report['server_exit'] or cancellations < args.runs:
        report.update(status='fail', observed_cancel_tasks=cancellations)
        save()
        raise RuntimeError('graceful shutdown or actual cancellation log evidence missing')
    report.update(status='pass', observed_cancel_tasks=cancellations)
    save()
    print(json.dumps({'status': report['status'], 'runs': args.runs, 'output': str(out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
