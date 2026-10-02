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
    sp.refinement_options.mode = mode
    sp.anneal_cycles = 64
    sp.num_instances = 8
    sp.num_workers = 2
    sp.deterministic_seed = True
    sp.random_seed = 731
    model = sa.SimAnneal(sp)
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
