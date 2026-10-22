#!/usr/bin/env python3
"""Summarize ESE_EXPERT_TRACE without adding overlapping timers together."""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import statistics


def summarize(path):
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for event in events:
        groups[event['scheduler'], event['pass']].append(event)
    passes = []
    history = defaultdict(lambda: deque(maxlen=3))
    frequency = defaultdict(Counter)
    predictions = defaultdict(Counter)
    early = Counter()
    lead_ms = []
    residency_recorded = True
    layer_waits = defaultdict(Counter)
    for _, items in groups.items():
        graphs = [x for x in items if x['event'] == 'graph']
        routes = sorted((x for x in items if x['event'] == 'route'), key=lambda x: x['layer'])
        residency_recorded = residency_recorded and all('resident_masks' in x for x in routes)
        stages = [x for x in items if x['event'] == 'stage']
        if not graphs or not routes:
            continue
        graph = graphs[-1]
        tokens = routes[0]['tokens']
        assert all(x['tokens'] == tokens for x in routes)
        row = {k: graph[k] for k in ('pass', 'host_schedule_ns', 'physical_read_bytes', 'splits')}
        row['tokens'] = tokens
        for key in ('elapsed_ns', 'route_wait_ns', 'lease_wait_ns', 'transfer_submit_ns', 'transfer_wait_ns', 'h2d_bytes'):
            row['stage_' + key] = sum(x[key] for x in stages)
        passes.append(row)
        if tokens == 1:
            actual_by_layer = {x['layer']: x for x in routes}
            for prediction in (x for x in items if x['event'] == 'prediction'):
                actual_route = actual_by_layer.get(prediction['layer'])
                if actual_route is None:
                    continue
                actual = set(actual_route['experts']) - {-1}
                actual_misses = {expert for expert, mask in zip(actual_route['experts'], actual_route['resident_masks'])
                                 if expert >= 0 and mask & 7 != 7}
                predicted = set(prediction['experts']) - {-1}
                candidates = {expert for expert, mask in zip(prediction['experts'], prediction['resident_masks'])
                              if expert >= 0 and mask & 7 != 7}
                early.update(actual=len(actual), predicted=len(predicted), matches=len(actual & predicted),
                             actual_misses=len(actual_misses), predicted_misses=len(candidates),
                             matched_misses=len(actual_misses & candidates))
                lead_ms.append((actual_route['at_ns'] - prediction['at_ns']) / 1e6)
            for stage in stages:
                layer_waits[stage['layer']].update({key: stage[key] for key in
                    ('elapsed_ns', 'lease_wait_ns', 'transfer_submit_ns', 'h2d_bytes')})
        for route in routes:
            layer = route['layer']
            width = route['top_k']
            for index in range(tokens):
                actual = set(route['experts'][index*width:(index+1)*width]) - {-1}
                actual_misses = {expert for expert, mask in zip(route['experts'][index*width:(index+1)*width],
                    route.get('resident_masks', [7]*len(route['experts']))[index*width:(index+1)*width]) if expert >= 0 and mask & 7 != 7}
                past = history[layer]
                candidates = {'last': past[-1] if past else set(),
                              'stable2': set.intersection(*list(past)[-2:]) if len(past) >= 2 else set(),
                              'stable3': set.intersection(*past) if len(past) >= 3 else set(),
                              'frequency': set(x for x, _ in frequency[layer].most_common(width))}
                if tokens == 1:
                    for name, predicted in candidates.items():
                        predictions[name].update(actual=len(actual), predicted=len(predicted),
                                                 matches=len(actual & predicted), actual_misses=len(actual_misses),
                                                 matched_misses=len(actual_misses & predicted))
                past.append(actual)
                frequency[layer].update(actual)
    decode = [row for row in passes if row['tokens'] == 1]
    keys = ('host_schedule_ns', 'physical_read_bytes', 'stage_elapsed_ns', 'stage_route_wait_ns',
            'stage_lease_wait_ns', 'stage_transfer_submit_ns', 'stage_transfer_wait_ns', 'stage_h2d_bytes')
    totals = {key: sum(row[key] for row in decode) for key in keys}
    result = {
        'trace': str(path.resolve()), 'graph_calls': len(passes), 'decode_calls': len(decode),
        'residency_recorded': residency_recorded,
        'timing_scope': 'Existing host scheduling only; no additional GPU synchronization. Stage sub-timers overlap GPU work. Physical reads are process-wide Linux read_bytes deltas, including any speculative I/O.',
        'decode_totals': totals,
        'decode_mean': {key: value / len(decode) for key, value in totals.items()} if decode else {},
        'decode_median_host_ms': statistics.median(row['host_schedule_ns'] / 1e6 for row in decode) if decode else None,
        'past_only_predictions': {name: {**counts, 'precision': counts['matches'] / max(1, counts['predicted']),
            'recall': counts['matches'] / max(1, counts['actual']),
            'miss_recall': counts['matched_misses'] / max(1, counts['actual_misses']) if residency_recorded else None} for name, counts in predictions.items()},
        'early_router': {**early, 'precision': early['matches'] / max(1, early['predicted']),
            'miss_precision': early['matched_misses'] / max(1, early['predicted_misses']),
            'miss_recall': early['matched_misses'] / max(1, early['actual_misses']),
            'median_lead_ms': statistics.median(lead_ms) if lead_ms else None},
        'decode_layers': {str(layer): dict(counts) for layer, counts in layer_waits.items()},
        'passes': passes,
    }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = summarize(args.trace)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('passes', 'decode_layers')}, indent=2))
