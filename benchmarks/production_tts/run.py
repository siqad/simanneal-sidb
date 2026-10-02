#!/usr/bin/env python3
"""Bounded production-solver benchmark. Run without competing CPU/GPU work."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('binary', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--jobs', type=int, default=64)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--seed', type=int, default=3280000003)
parser.add_argument('--qualification', action='store_true')
parser.add_argument('--replay-of', help='SHA256 of prior rows when deliberately replaying the same seeds')
args = parser.parse_args()
if not 1 <= args.jobs <= 128 or args.workers <= 0:
    parser.error('jobs must be 1..128 and workers must be positive')
if args.qualification and args.jobs != 1:
    parser.error('qualification requires --jobs 1 to preserve disjoint seed intervals')
if args.replay_of and (len(args.replay_of) != 64 or any(c not in '0123456789abcdef' for c in args.replay_of)):
    parser.error('--replay-of must be a lowercase SHA256 digest')
binary = args.binary.resolve()
args.output.mkdir(parents=True, exist_ok=False)
cases = json.loads(Path(__file__).with_name('cases.json').read_text())
arms = ['legacy', 'fixed_portable', 'fixed_blas', 'shared_blas', 'legacy_tuned']
started = time.monotonic()
case_stride = 100000 if args.qualification else 1000000
rows = []
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', GOTO_NUM_THREADS='1')
plan = dict(jobs=args.jobs, workers=args.workers, seed=args.seed, case_stride=case_stride, arms=arms,
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            cases_sha256=hashlib.sha256(Path(__file__).with_name('cases.json').read_bytes()).hexdigest(),
            timeout_seconds=120, campaign_seconds=1800,
            replay_of_rows_sha256=args.replay_of, independent_seed_cohort=not bool(args.replay_of),
            comparison='Matched restart intervals and four original schedules; legacy_tuned uses prescribed longer cycles and hop factor 5 on large layouts. ' + ('Same-seed replay of the recorded prior cohort. ' if args.replay_of else 'Fresh seeds. ') + 'Large targets are not certified ground states.',
            timing='Prepared-input complete production compute job; coordinate/SimParams allocation, solver construction, search, validated export and solver-owned cleanup included; file parsing, process start, retained output and stdout excluded.')
(args.output/'plan.json').write_text(json.dumps(plan, indent=2))
for block in range((args.jobs+7)//8):
    for arm in arms[block%len(arms):]+arms[:block%len(arms)]:
        for i, case in enumerate(cases):
            requests = []
            for j in range(block*8, min(args.jobs, block*8+8)):
                q = dict(case)
                q.update(id=f'{i}:{j}:{arm}', seed=args.seed+i*case_stride+j*4096, workers=args.workers,
                         search_profile='legacy' if arm.startswith('legacy') else 'optimized',
                         backend='portable' if arm in ['legacy', 'legacy_tuned', 'fixed_portable'] else 'openblas',
                         refinement='none' if arm.startswith('legacy') else 'shared' if arm == 'shared_blas' and case['cluster_size']==10 else f"k{case['cluster_size']}",
                         refinement_candidates=8, refinement_rounds=case['job_post_rounds'],
                         transient_domain_mask=not arm.startswith('legacy'), probability_shortcuts=not arm.startswith('legacy'))
                if arm == 'legacy_tuned' and not case['certified_exact_target']:
                    q.update(cycles=max(512,q['cycles']), hop_factor=5)
                if args.qualification:
                    q.update(restarts=16, cycles=32, refinement_rounds=1)
                if q['seed'] < 0 or q['seed']+q['restarts'] >= 2**32:
                    raise ValueError('Restart seeds must fit uint32_t')
                requests.append(q)
            stem = f'b{block}_{i}_{arm}'
            request_path = args.output/(stem+'_requests.json')
            request_path.write_text(json.dumps({'requests':requests}))
            remaining = 1800 - (time.monotonic()-started)
            if remaining <= 0:
                raise RuntimeError('Campaign cap exceeded')
            begin = time.perf_counter()
            process = subprocess.Popen([str(binary), str(request_path)], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
            try:
                stdout, stderr = process.communicate(timeout=min(120, remaining))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
                (args.output/(stem+'_timeout.txt')).write_text(stdout+'\n'+stderr)
                raise RuntimeError('Process group exceeded its process or campaign time limit')
            (args.output/(stem+'_raw.txt')).write_text(stdout)
            (args.output/(stem+'_meta.json')).write_text(json.dumps(dict(returncode=process.returncode,
                process_wall_ms=(time.perf_counter()-begin)*1000, stderr=stderr)))
            if process.returncode != 0:
                raise RuntimeError(f'{stem}: solver exited {process.returncode}: {stderr}')
            output = [json.loads(x) for x in stdout.splitlines() if x.startswith('{')]
            if len(output) != len(requests):
                raise RuntimeError(f'{stem}: expected {len(requests)} rows, got {len(output)}')
            for result, request in zip(output, requests):
                if result.get('id') != request['id']:
                    raise RuntimeError(f'{stem}: output ID does not match request')
                result.update(case=case['name'], arm=arm, seed=request['seed'], block=block)
                rows.append(result)
            (args.output/'rows.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        print('Finished block',block,arm,'jobs',len(rows),flush=True)
groups = defaultdict(list)
for row in rows:
    groups[row['case'], row['arm']].append(row)
summary = []
for case in cases:
    for arm in arms:
        group = groups[case['name'],arm]
        mean = statistics.mean(float(r['total_ms']) for r in group)
        targets = {}
        for name, target in case['targets'].items():
            hits = sum(r['valid'] in [True,'true','1'] and float(r['energy']) <= target+1e-8 for r in group)
            p = hits/len(group)
            tts = None if not hits else mean if hits == len(group) else mean*max(1, math.log(.003)/math.log1p(-p))
            targets[name] = dict(energy=target, hits=hits, jobs=len(group), tts997_ms=tts)
        summary.append(dict(case=case['name'], arm=arm, mean_ms=mean,
                            certified_exact_target=case['certified_exact_target'], targets=targets))
(args.output/'summary.json').write_text(json.dumps(summary,indent=2))
print('Complete:',len(rows),'jobs',flush=True)
