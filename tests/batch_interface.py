"""Run with the Python interpreter matching the freshly built SWIG module."""
import json
import os
from pathlib import Path
import subprocess
import sys

source = Path(__file__).resolve().parents[1]
environment = dict(os.environ, PYTHONPATH=str(Path(sys.argv[1]).resolve()))
job = {"id": "first", "points": [[0.0, 0.0], [7.68, 0.0], [0.0, 7.68], [7.68, 7.68]],
       "mu": -.32, "population_backend": "portable", "num_workers": 1, "seed": 731}
lines = [json.dumps(job), '{"points": [], "num_instances": 0}',
         '{"points": [], "seed": 1, "seed": 2}', '{"id": NaN, "points": []}', '{"id": 1e999, "points": []}']
invalid = [({key: value}, key + " must be a finite number")
           for key in ["mu", "epsilon_r", "lambda_tf"]
           for value in [True, False, "1.0", None, 10 ** 400]]
invalid += [({"points": [[value, 0]]}, "points must be a finite number")
            for value in [True, "0", None, 10 ** 400]]
invalid += [({"external_potential": value}, "external_potential must have one value per point")
            for value in [[], [0], "0000", None, {}]]
invalid += [({"external_potential": [value, 0, 0, 0]}, "external_potential must be a finite number")
            for value in [True, False, "0", None, 10 ** 400]]
invalid += [({"population_backend": value}, "population_backend must be one of: ")
            for value in ["unknown", [], {}, None, True, 1]]
lines += [json.dumps(dict(job, **fields)) for fields, _ in invalid]
lines.append(json.dumps(dict(job, id="last")))
process = subprocess.run([sys.executable, str(source / "examples/simulate_batch.py"), "-"],
                         input="\n".join(lines), capture_output=True, text=True,
                         env=environment, timeout=30)
assert process.returncode == 1, process.stderr
results = [json.loads(line) for line in process.stdout.splitlines()]
assert [r["ok"] for r in results] == [True] + [False] * (4 + len(invalid)) + [True]
assert results[0]["results"] == results[-1]["results"]
assert results[0]["effective"] == {"anneal_cycles": 256, "num_instances": 8,
                                  "hop_attempt_factor": 2, "num_workers": 1}
assert results[0]["stats"] == results[-1]["stats"]
assert results[0]["results"]
assert "duplicate key" in results[2]["error"]
assert all("nonfinite" in results[i]["error"] for i in [3, 4])
for result, (_, message) in zip(results[5:-1], invalid):
    assert message in result["error"], result
print("Batch requests, Auto metadata, seeded independence, and error recovery passed")
