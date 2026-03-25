from clspde.solution import Solution
from clspde.utils import plot, eval_dict
from clspde.prepare import from_file
#from clspde.basis import Basis
import copy
import itertools
import numpy as np
import copy

import yaml
import pickle as pkl

'''
def sol_eval(sol, ts=ts, xs=xs):
    res_array = np.zeros((len(ts), len(xs)))
    for i in range(len(ts)):
        t = ts[i]
        for j in range(len(xs)):
            x = xs[j]
            res_array[i, j] = sol.eval(np.array([t, x]), [0, 0])

    return res_array
'''

settings_filename = "settings/simplest_mfg.yaml"
settings, sol, iteration_dict = from_file(settings_filename)

n = 20
ts = np.linspace(settings['MODEL']["area_lims"][0, 0], settings['MODEL']["area_lims"][0, 1] - 0.00001, n)

k = 50
for j in range(k):
    prev_coefs = copy.deepcopy(sol.cells_coefs)
    #prev_eval = sol_eval(sol)
    A, b = sol.global_solve(
        solver="np",
        #svd_threshold=1e-8,
        alpha=1e-4,
        **iteration_dict,
    )
    speed = 1
    sol.cells_coefs = (1-speed)*prev_coefs + speed*sol.cells_coefs 
    print(j,' | ', np.max(np.abs(prev_coefs - sol.cells_coefs)),' | ',) #np.max(np.abs(prev_eval - sol_eval(sol))), ' | ', np.max(np.abs(A @ sol.cells_coefs.ravel() - b)))

plot(sol)

params_to_save = copy.deepcopy(params)
params_to_save.pop("basis", None)
params_to_save["coefs"] = sol.cells_coefs

dump_pars(pars_to_save)

points = []
vals = []
for t in ts:
    points.append([t])
    vals.append([sol.eval(np.array([t]), [0], 1)])

import matplotlib.pyplot as plt
plt.plot(points, vals)
plt.show()

out_dict = {'points':points, 'data':vals}
with open('colloc_solution_I.pkl', 'wb') as out_file:
    pkl.dump(out_dict, out_file)

