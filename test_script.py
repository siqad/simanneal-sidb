#/usr/bin/env/python
# encoding: utf-8

import simanneal
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
sp.set_db_locs(or_00())

sa = simanneal.SimAnneal(sp)
sa.invokeSimAnneal()

results = sa.suggested_gs_results()
print(results)
