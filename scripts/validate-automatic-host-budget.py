#!/usr/bin/env python3
"""Exercise automatic strict-RAM startup, cold inference and serialized reservations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-layers', type=int, default=99)
    parser.add_argument('--server-arg', action='append', default=[])
    args = parser.parse_args()
    server = args.server.resolve(strict=True)
    model = args.model.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    command = [str(server), '-m', str(model), '--memory-policy', 'auto', '--max-ram', '4GiB',
               '--max-context', '1024', '--min-kv-quality', 'f16', '-c', '1024',
               '-ngl', str(args.gpu_layers), '--host', '127.0.0.1', '--port', str(port),
               '--no-warmup', '-no-ooae', '-t', '4', '-tb', '4'] + args.server_arg
    environment = os.environ.copy()
    if args.gpu_layers == 0:
        environment['CUDA_VISIBLE_DEVICES'] = ''
    report = {'status': 'running', 'argv': command,
              'server_sha256': hashlib.file_digest(server.open('rb'), 'sha256').hexdigest(),
              'model_sha256': hashlib.file_digest(model.open('rb'), 'sha256').hexdigest(),
              'environment': {k: v for k, v in environment.items()
                              if k.startswith(('CUDA_', 'LD_', 'ESE_', 'GGML_'))}}
    request_number = 0

    def request(path, payload=None):
        nonlocal request_number
        req = urllib.request.Request(f'http://127.0.0.1:{port}{path}',
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.load(response)
        request_number += 1
        (out / f'{request_number:04d}-response.json').write_text(json.dumps(
            {'path': path, 'payload': payload, 'response': result}, indent=2)+'\n')
        return result

    process = None
    try:
        with (out / 'server.log').open('w') as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic()+180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'server exited during startup: {process.returncode}')
                try:
                    request('/health')
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RuntimeError('startup timeout')
                    time.sleep(.1)
            payload = dict(prompt='The capital of France is', n_predict=16, n_probs=10,
                           temperature=0, seed=4242, ignore_eos=True, cache_prompt=False)
            first = request('/completion', payload)
            second = request('/completion', payload)
            if first['tokens_predicted'] != 16 or second['tokens_predicted'] != 16:
                raise RuntimeError('incomplete generation')
            for field in ('content', 'completion_probabilities'):
                if field not in first or first[field] != second[field]:
                    raise RuntimeError(f'cold replay mismatch: {field}')
            snapshot = request('/v1/ese/resources')
            # Read the finalized startup receipt, rather than a provisional solve.
            text = (out / 'server.log').read_text()
            plans = [json.loads(line.split('resource_plan: ', 1)[1])
                     for line in text.splitlines() if line.startswith('resource_plan: ')]
            if len(plans) != 1:
                raise RuntimeError('missing or duplicated finalized resource plan')
            plan = plans[0]
            if plan['ram_capacity_bytes'] != 4*1024**3 or plan['aux_ram_bytes'] % 1024**2:
                raise RuntimeError('RAM capacity or whole-MiB cache receipt mismatch')
            if plan.get('host_context_metadata_bytes', 0) <= 0:
                raise RuntimeError('native context reservation is missing')
            reserved = sum(plan[key] for key in ('expert_ram_bytes', 'aux_ram_bytes', 'io_staging_bytes',
                'qwen4exp_ple_ram_bytes', 'host_graph_metadata_bytes', 'host_context_metadata_bytes', 'host_graph_workspace_bytes'))
            weights = plan['ram_planned_bytes']-reserved
            host = [p for p in snapshot['memory_inventory']['pools'] if p['device_id'] == -1]
            tracked_weights = sum(p['bytes'] for p in host if p['category'] == 0)
            tracked_fixed = sum(p['bytes'] for p in host if p['category'] not in (0, 1, 2, 6, 11))
            tracked_ple = sum(p['bytes'] for p in host if p['category'] == 11)
            tracked_kv = sum(p['bytes'] for p in host if p['category'] in (2, 6))
            admitted = (max(weights, tracked_weights) + max(plan['qwen4exp_ple_ram_bytes'], tracked_ple)
                        + max(plan['host_graph_metadata_bytes']+plan['host_context_metadata_bytes'], tracked_fixed)
                        + tracked_kv + plan['expert_ram_bytes']+plan['aux_ram_bytes']+plan['io_staging_bytes'])
            if weights < 0 or admitted > plan['ram_capacity_bytes']:
                raise RuntimeError(f'native and optional reservations exceed RAM: {admitted}')
            if not snapshot['safe_point']:
                raise RuntimeError('inference did not drain')
            report.update(status='pass', plan=plan, admitted_host_bytes=admitted,
                          output_sha256=hashlib.sha256(first['content'].encode()).hexdigest(),
                          cold_replay_probabilities_exact=True)
    except Exception as error:
        report.update(status='fail', error=str(error))
    finally:
        if process is not None:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            report['server_returncode'] = process.returncode
            if process.returncode != 0:
                report.update(status='fail')
        (out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
