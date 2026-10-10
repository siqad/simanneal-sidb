"""Stream independent native SimAnneal jobs without reloading Python for every job."""
import argparse
import contextlib
import json
import math
import sys

try:
    import simanneal as sa
except ModuleNotFoundError as error:
    if error.name != "simanneal":
        raise
    from pysimanneal import simanneal as sa


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("nonfinite JSON number: " + value)


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite JSON number: " + value)
    return result


def simulate(job):
    allowed = {"id", "points", "mu", "epsilon_r", "lambda_tf", "external_potential",
               "seed", "anneal_cycles", "num_instances", "hop_attempt_factor",
               "num_workers", "population_backend"}
    if not isinstance(job, dict) or set(job) - allowed:
        raise ValueError("request must be an object with documented keys")
    points = job["points"]
    if not isinstance(points, list) or not points or any(not isinstance(p, list) or len(p) != 2 for p in points):
        raise ValueError("points must contain nonempty [x, y] pairs in angstroms")
    params = sa.SimParams()
    params.set_db_locs(points)
    params.mu = job.get("mu", params.mu)
    params.eps_r = job.get("epsilon_r", params.eps_r)
    params.debye_length = job.get("lambda_tf", params.debye_length)
    params.set_v_ext(job.get("external_potential", [0.0] * len(points)))
    params.set_fixed_charges([], [], [], [])
    for key, automatic in [("anneal_cycles", sa.AutoAnnealCycles),
                           ("num_instances", sa.AutoInstances),
                           ("hop_attempt_factor", sa.AutoHopAttempts), ("num_workers", 0)]:
        if key in job:
            value = job[key]
            if value == "auto":
                value = automatic
            if type(value) is not int:
                raise ValueError(key + " must be an integer or auto")
            setattr(params, key, value)
    backends = {"auto": sa.PopulationBackend_Auto, "portable": sa.PopulationBackend_Portable,
                "accelerate": sa.PopulationBackend_Accelerate, "openblas": sa.PopulationBackend_OpenBLAS,
                "openblas_symmetric": sa.PopulationBackend_OpenBLASSymmetric}
    params.population_backend = backends[job.get("population_backend", "auto")]
    if "seed" in job:
        if type(job["seed"]) is not int or not 0 <= job["seed"] <= 4294967295:
            raise ValueError("seed must be an integer in [0,4294967295]")
        params.random_seed = job["seed"]
        params.deterministic_seed = True
    model = sa.SimAnneal(params)
    try:
        model.invokeSimAnneal()
        effective, stats = model.effectiveParams(), model.searchStats()
        return {"results": [{"config": r.config, "energy": r.energy}
                            for r in model.suggested_gs_results()],
                "effective": {key: getattr(effective, key) for key in
                              ["anneal_cycles", "num_instances", "hop_attempt_factor", "num_workers"]},
                "stats": {key: getattr(stats, key) for key in
                          ["executed_restarts", "repair_attempts", "repair_budget_exhaustions", "singleton_used"]}}
    finally:
        del model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON Lines request file, or - for stdin")
    args = parser.parse_args()
    failures = False
    source = contextlib.nullcontext(sys.stdin) if args.input == "-" else open(args.input, encoding="utf-8")
    with source as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            identity = line_number
            try:
                job = json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant, parse_float=finite_float)
                if isinstance(job, dict):
                    identity = job.get("id", identity)
                result = dict(id=identity, ok=True, **simulate(job))
            except Exception as error:
                failures = True
                result = {"id": identity, "ok": False, "error": str(error)}
            print(json.dumps(result, allow_nan=False), flush=True)
    return int(failures)


if __name__ == "__main__":
    sys.exit(main())
