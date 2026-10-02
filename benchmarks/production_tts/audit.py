#!/usr/bin/env python3
"""Independent FP64 physical audit after timing. Requires NumPy."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('output', type=Path)
args = parser.parse_args()
cases = {c['name']:c for c in json.loads(Path(__file__).with_name('cases.json').read_text())}
models, seen = {}, set()
maximum_error = 0.0
for line in (args.output/'rows.jsonl').read_text().splitlines():
    row = json.loads(line)
    if row['valid'] not in [True, 'true', '1']:
        continue
    name = row['case']
    q = np.asarray(row['config'], dtype=np.float64)
    key = (name, tuple(q))
    if key in seen:
        continue
    seen.add(key)
    c = cases[name]
    if name not in models:
        xy = np.array(c['points']) * 1e-10
        d = np.sqrt(np.sum((xy[:,None,:]-xy[None,:,:])**2, axis=2))
        np.fill_diagonal(d, 1)
        A = 1.602e-19 / (4*3.14159*8.854e-12*c['epsilon_r']) * np.exp(-d/(c['lambda_tf']*1e-9))/d
        np.fill_diagonal(A, 0)
        b = np.array(c['external_potential']) + np.array(c['fixed_potential'])
        models[name] = A,b
    A,b = models[name]
    assert len(q)==len(A) and np.isin(q, [-1,0,1]).all(), name
    v = -b-A@q
    mu, eta, eps = c['mu'], .59, 1e-6
    population = ((q==-1)&(v+mu<eps)) | ((q==1)&(v+mu-eta>-eps)) | ((q==0)&(v+mu>-eps)&(v+mu-eta<eps))
    assert population.all(), (name,'population',q)
    delta = -v[:,None]+v[None,:]-A
    assert np.all(delta[q[:,None]<q[None,:]] >= -eps-1e-12), (name,'hop')
    energy = float(b@q + .5*q@A@q)
    error = abs(energy-float(row['energy']))
    maximum_error = max(maximum_error,error)
    assert error <= 1e-10, (name,energy,row['energy'])
result = dict(unique_states=len(seen), rejects=0, maximum_energy_error=maximum_error,
              rows_sha256=hashlib.sha256((args.output/'rows.jsonl').read_bytes()).hexdigest())
(args.output/'audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
