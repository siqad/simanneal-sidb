#!/usr/bin/env python3
"""Independently check the implemented FP64 model, retaining its legacy constants."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def audit_rows(rows, cases):
    """Check every claimed-valid row; count rejected rows even under python -O."""
    models, seen = {}, {}
    maximum_error, checked_rows = 0.0, 0
    failures = []
    for index, row in enumerate(rows):
        if row.get('valid') not in [True, 'true', '1']:
            continue
        checked_rows += 1
        errors = []
        try:
            name = row['case']
            c = cases[name]
            q = np.asarray(row['config'], dtype=np.float64)
            if q.shape != (len(c['points']),) or not np.isin(q, [-1, 0, 1]).all():
                raise ValueError('invalid charge vector')
            key = (name, tuple(q))
            if key not in seen:
                if name not in models:
                    xy = np.array(c['points']) * 1e-10
                    d = np.sqrt(np.sum((xy[:, None, :] - xy[None, :, :])**2, axis=2))
                    np.fill_diagonal(d, 1)
                    # Intentionally preserve the solver's rounded model constants.
                    # This checks the implementation, not the constants' accuracy.
                    A = 1.602e-19 / (4*3.14159*8.854e-12*c['epsilon_r']) * np.exp(-d/(c['lambda_tf']*1e-9))/d
                    np.fill_diagonal(A, 0)
                    b = np.array(c['external_potential']) + np.array(c['fixed_potential'])
                    models[name] = A, b
                A, b = models[name]
                v = -b - A @ q
                mu, eta, eps = c['mu'], .59, 1e-6
                population = ((q == -1) & (v+mu < eps)) | ((q == 1) & (v+(mu-eta) > -eps)) | ((q == 0) & (v+mu > -eps) & (v+(mu-eta) < eps))
                if not population.all():
                    errors.append('population')
                delta = -v[:, None] + v[None, :] - A
                if not np.all(delta[q[:, None] < q[None, :]] >= -eps-1e-12):
                    errors.append('hop')
                energy = float(b @ q + .5 * q @ A @ q)
                seen[key] = energy, tuple(errors)
            energy, state_errors = seen[key]
            errors = list(state_errors)
            reported = float(row['energy'])
            if not math.isfinite(energy) or not math.isfinite(reported):
                errors.append('nonfinite energy')
            else:
                error = abs(energy-reported)
                maximum_error = max(maximum_error, error)
                if error > 1e-10:
                    errors.append('energy mismatch')
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            errors.append(str(error))
        if errors:
            failures.append(dict(row=index, case=row.get('case'), errors=errors))
    return dict(unique_states=len(seen), checked_rows=checked_rows,
                rejects=len(failures), failures=failures,
                maximum_energy_error=maximum_error,
                scope='Independent implementation check of the solver model, including legacy rounded constants; not validation of physical constants.')


def main():
    """Write the audit, including failures, and return a failing exit status."""
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    cases = {c['name']: c for c in json.loads(Path(__file__).with_name('cases.json').read_text())}
    raw = (args.output/'rows.jsonl').read_bytes()
    result = audit_rows([json.loads(line) for line in raw.splitlines()], cases)
    result['rows_sha256'] = hashlib.sha256(raw).hexdigest()
    (args.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    return 1 if result['rejects'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
