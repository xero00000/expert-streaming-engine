#!/usr/bin/env python3
"""Screen synchronous/worker expert streaming with actual uploads and bounded leases.

Use an existing supported MoE GGUF. This warm-source same-prompt screen is not
large-model performance promotion or cross-placement numerical certification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-sets', default='0;0,2;0,1,2', help='semicolon-separated CUDA visible device sets')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--predict', type=int, default=128)
    args = parser.parse_args()
    if args.rounds < 1 or args.predict < 16:
        parser.error('rounds must be positive and predict must be at least 16')
    cli, model = args.cli.resolve(strict=True), args.model.resolve(strict=True)
    gpu_sets = args.gpu_sets.split(';')
    if any(not lane or any(not device for device in lane.split(',')) for lane in gpu_sets):
        parser.error('GPU sets must contain nonempty device identifiers')
    report = {
        'scope': __doc__, 'complete': False, 'rows': [],
        'cli': str(cli), 'model': str(model),
        'cli_sha256': hashlib.sha256(cli.read_bytes()).hexdigest(),
        'libraries_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (cli.parents[1] / 'src/libllama.so', cli.parents[1] / 'ggml/src/libggml.so')},
        'predict': args.predict, 'rounds': args.rounds,
        'cache_control': 'no OS/device cache eviction; engine reloaded for every run',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    save()
    references = {}
    for trial in range(args.rounds):
        for visible in gpu_sets:
            configs = [(0, 'pread'), (2, 'pread'), (0, 'io_uring'), (2, 'io_uring')]
            if trial % 2:
                configs.reverse()
            for workers, backend in configs:
                command = [str(cli), '-m', str(model), '-p', 'The capital of France is',
                    '-n', str(args.predict), '-c', '2048', '-s', '4242', '--temp', '0',
                    '--no-display-prompt', '--simple-io', '--no-warmup', '-t', '4',
                    '-ngl', '99', '-ub', '1', '-sm', 'layer', '-ts', ','.join('1' for _ in visible.split(',')),
                    '-ot', '.ffn_.*_exps.=CPU', '--expert-ram-cache-mib', '8',
                    '--expert-ram-staging-mib', '2', '--expert-sidecar-only', '--defer-experts',
                    '--expert-storage-backend', backend, '--expert-vram-cache-mib', '8',
                    '--expert-vram-reserve-mib', '64', '--expert-cache-min-observations', '2',
                    '--expert-load-workers', str(workers)]
                environment = dict(os.environ)
                environment.update(CUDA_VISIBLE_DEVICES=visible, GGML_CUDA_NO_PINNED='1')
                environment.pop('LD_PRELOAD', None)
                started = time.monotonic()
                result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=120)
                stats = [json.loads(value) for value in re.findall(r'expert_cache_stats:\s*(\{[^\n]*\})', result.stderr)]
                queues = [json.loads(value) for value in re.findall(r'expert_load_queue_stats:\s*(\{[^\n]*\})', result.stderr)]
                timing = re.search(r'(?<!prompt )eval time\s*=\s*([0-9.]+) ms /\s*(\d+) tokens', result.stderr)
                row = {'trial': trial, 'visible_devices': visible, 'workers': workers, 'backend': backend,
                    'command': command, 'environment': {'CUDA_VISIBLE_DEVICES': visible, 'GGML_CUDA_NO_PINNED': '1'},
                    'exit': result.returncode, 'wall_s': time.monotonic() - started,
                    'output_sha256': hashlib.sha256(result.stdout.encode()).hexdigest(),
                    'decode_tps': int(timing[2]) * 1000 / float(timing[1]) if timing else None,
                    'stats': stats, 'queues': queues, 'validated': False}
                report['rows'].append(row)
                save()
                if result.returncode:
                    raise RuntimeError(result.stderr[-4000:])
                if not timing or int(timing[2]) < args.predict - 1:
                    raise RuntimeError('request did not produce the required sustained decode')
                reference = references.setdefault(visible, row['output_sha256'])
                if reference != row['output_sha256']:
                    raise RuntimeError('worker/backend output mismatch at identical placement')
                ram = next(item for item in stats if item.get('level') == 'ram')
                total = next(item for item in stats if item.get('level') == 'vram-total')
                devices = [item for item in stats if item.get('level') == 'vram']
                if total['lease_uploads'] == 0 or total['forced_fallbacks'] != 0:
                    raise RuntimeError('strict GPU streaming did not execute without fallback')
                if len(devices) != len(visible.split(',')) or any(item['resident_bytes'] > item['capacity_bytes'] for item in devices):
                    raise RuntimeError('participating device count or VRAM bound failed')
                if ram['peak_resident_bytes'] > ram['capacity_bytes'] or ram['peak_staging_used_bytes'] > ram['staging_capacity_bytes']:
                    raise RuntimeError('RAM/staging bound failed')
                if any(ram[key] for key in ('active_leases', 'active_loads', 'reserved_bytes', 'staging_used_bytes', 'failed_loads')):
                    raise RuntimeError('cache did not drain cleanly')
                if workers and (not queues or queues[-1]['submitted'] == 0 or
                        queues[-1]['outstanding'] or queues[-1]['outstanding_bytes'] or
                        queues[-1]['peak_outstanding_bytes'] > queues[-1]['byte_limit'] or ram['peak_active_loads'] < 2):
                    raise RuntimeError('worker concurrency or queue bounds not proven')
                row['validated'] = True
                save()
                print(json.dumps({key: row[key] for key in ('trial', 'visible_devices', 'workers', 'backend', 'decode_tps')}), flush=True)
    report['summary'] = [
        {'visible_devices': visible, 'backend': backend, 'workers': workers,
         'median_decode_tps': statistics.median(row['decode_tps'] for row in report['rows']
            if (row['visible_devices'], row['backend'], row['workers']) == (visible, backend, workers))}
        for visible in gpu_sets for backend in ('pread', 'io_uring') for workers in (0, 2)]
    report['complete'] = True
    save()
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
