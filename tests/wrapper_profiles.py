"""Run against a freshly built SWIG module: python wrapper_profiles.py BUILD_DIR."""
import gc
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import simanneal as sa

for mode in [sa.Mode_Disabled, sa.Mode_K6, sa.Mode_K10, sa.Mode_SharedK10]:
    sp = sa.SimParams()
    sp.set_db_locs([[0.,0.], [7.68,0.], [0.,7.68], [7.68,7.68], [15.36,0.], [15.36,7.68]])
    sp.set_v_ext([0.] * 6)
    sp.set_fixed_charges([], [], [], [])
    assert sp.search_profile == sa.SearchProfile_Optimized
    assert sp.population_probability_cache is False
    sp.population_probability_cache = True
    assert sp.population_probability_cache is True
    sp.refinement_options.mode = mode
    sp.anneal_cycles = 64
    sp.num_instances = 8
    sp.num_workers = 2
    sp.deterministic_seed = True
    sp.random_seed = 731
    model = sa.SimAnneal(sp)
    assert model.effectiveParams().population_probability_cache is True
    model.invokeSimAnneal()
    results = model.suggested_gs_results()
    assert results and all(math.isfinite(r.energy) and len(r.config)==6 for r in results)
    points = [(0.,0.), (7.68,0.), (0.,7.68), (7.68,7.68), (15.36,0.), (15.36,7.68)]
    for result in results:
        energy = 0.0
        for i in range(6):
            for j in range(i):
                r = math.dist(points[i],points[j])*1e-10
                energy += result.config[i]*result.config[j]*1.602e-19/(4*3.14159*8.854e-12*5.6)*math.exp(-r/5e-9)/r
        assert abs(result.energy-energy)<1e-12
    assert model.searchStats().executed_restarts == 8
    try:
        second = sa.SimAnneal(sp)
    except RuntimeError as error:
        assert 'one SimAnneal model' in str(error)
    else:
        raise AssertionError('Overlapping model accepted')
    expected = [(tuple(r.config), r.energy) for r in results]
    del model
    gc.collect()
    # Python parameters remain reusable, and models own their copied storage.
    model = sa.SimAnneal(sp)
    del sp
    gc.collect()
    model.invokeSimAnneal()
    assert [(tuple(r.config), r.energy) for r in model.suggested_gs_results()] == expected
    del model
    gc.collect()

sp = sa.SimParams()
sp.set_db_locs([[0.,0.]])
sp.set_v_ext([0.])
sp.set_fixed_charges([], [], [], [])
sp.search_profile = sa.SearchProfile_Optimized
model = sa.SimAnneal(sp)
model.invokeSimAnneal()
assert model.searchStats().singleton_used
assert model.searchStats().executed_restarts == 0
assert model.suggested_gs_results()[0].config == [-1]
del model
gc.collect()
sp.num_instances = 0
try:
    model = sa.SimAnneal(sp)
except RuntimeError:
    pass
else:
    raise AssertionError('Invalid restart count accepted')
print('SWIG profiles, FP64 results, metadata, lifecycle, and exceptions passed')

# Auto requests survive assignment and model ownership of copied parameters.
sp = sa.SimParams()
sp.set_db_locs([[0., 0.], [7.68, 0.], [15.36, 0.]])
sp.mu = -.32
sp.population_backend = sa.PopulationBackend_Portable
sp.num_workers = 1
sp.deterministic_seed = True
sp.random_seed = 731
assert (sp.anneal_cycles, sp.num_instances, sp.hop_attempt_factor) == (-1, -2, -1)
model = sa.SimAnneal(sp)
effective = model.effectiveParams()
assert (effective.anneal_cycles, effective.num_instances, effective.hop_attempt_factor) == (256, 8, 2)
assert (effective.requested_anneal_cycles, effective.requested_instances, effective.requested_hop_attempt_factor) == (-1, -2, -1)
assert effective.budget_auto_selected
sp.anneal_cycles = 256
sp.num_instances = 8
sp.hop_attempt_factor = 2
assert model.effectiveParams().requested_instances == -2
model.invokeSimAnneal()
expected = [(tuple(r.config), r.energy) for r in model.suggested_gs_results()]
assert model.searchStats().executed_restarts == 8
del effective, model
gc.collect()
model = sa.SimAnneal(sp)
del sp
gc.collect()
assert not model.effectiveParams().budget_auto_selected
model.invokeSimAnneal()
assert [(tuple(r.config), r.energy) for r in model.suggested_gs_results()] == expected
assert model.searchStats().executed_restarts == 8
del model
gc.collect()
print('SWIG Auto assignment, request metadata, lifetime, and seeded equivalence passed')

for mu in [-.32, -.25, -.20]:
    sp = sa.SimParams()
    sp.set_db_locs([[0., 0.], [7.68, 0.], [15.36, 0.]])
    sp.mu = mu
    sp.num_workers = 1
    model = sa.SimAnneal(sp)
    effective = model.effectiveParams()
    assert (effective.anneal_cycles, effective.num_instances, effective.hop_attempt_factor) == (256, 8, 2)
    assert effective.budget_auto_selected
    del effective, model
    gc.collect()
    sp.anneal_cycles, sp.num_instances, sp.hop_attempt_factor = 1000, 11, 3
    model = sa.SimAnneal(sp)
    effective = model.effectiveParams()
    assert (effective.anneal_cycles, effective.num_instances, effective.hop_attempt_factor) == (1000, 11, 3)
    assert not effective.budget_auto_selected
    del effective, model
    gc.collect()
print('SWIG mu-range Auto budgets and explicit overrides passed')

for count in [35, 36, 62, 63]:
    for eps in [5.6, 10.]:
        sp = sa.SimParams()
        sp.set_db_locs([[i*10000., 0.] for i in range(count)])
        sp.mu = -.20
        sp.eps_r = eps
        sp.debye_length = 10. if eps == 10. else 5.
        sp.num_workers = 1
        model = sa.SimAnneal(sp)
        effective = model.effectiveParams()
        expected = (256, 64, 2) if count == 35 else (512, 128, 2) if count <= 62 else (10000, 128, 5)
        assert (effective.anneal_cycles, effective.num_instances, effective.hop_attempt_factor) == expected
        assert effective.budget_auto_selected == (count <= 62)
        assert effective.requested_instances == -2
        del effective, model
        gc.collect()
print('SWIG larger and wider Auto qualification boundaries passed')
