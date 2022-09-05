#/usr/bin/env/python
# encoding: utf-8

# this import assumes that a pysimanneal directory containing __init__.py,
# simanneal.py, and the compiled simanneal library (_simanneal.so for Linux or
# _simanneal.pyd for Windows) are present.
from pysimanneal import simanneal
import numpy as np

def or_00():
    return [[-30.72, -30.72],
            [-23.04, -23.04],
            [-7.68, -23.04],
            [0, -30.72],
            [-15.36, -7.68],
            [-15.36, 2.25],
            [-15.36, 17.61]]

sp = simanneal.SimParams()

# set DB locations
sp.set_db_locs(or_00())

# set other simulation parameters, refer to the SimParams struct in simanneal.h
# for all parameters that can be changed.
sp.mu = -0.28

# use SimParams.set_v_ext if you want to set external potentials; this can be 
# omitted if external potentials are zero.
sp.set_v_ext(np.zeros(len(sp.db_locs))) 

# use SimParams.set_fixed_charges to add fixed charge defects.
#fc_locs = [[0, 0, -4]]  # list of 3d eucl coords in angstroms
#fc_c = [-1]             # list of corresponding charges
#fc_eps = [10.6]         # list of relative permittivities
#fc_lambdas = [5.9]      # list of debye lengths
#sp.set_fixed_charges(fc_locs, fc_c, fc_eps, fc_lambdas);

sa = simanneal.SimAnneal(sp)
sa.invokeSimAnneal()

results = sa.suggested_gs_results()
print(results)
