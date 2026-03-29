from clspde.utils import plot
from clspde.prepare import from_file
import itertools
import numpy as np
import copy
import yaml
import pickle as pkl


settings_filename = "settings/simplest_mfg.yaml"
settings, sol, iteration_dict = from_file(settings_filename)

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
    print(j,' | ', np.max(np.abs(prev_coefs - sol.cells_coefs)),' | ',)
plot(sol)

#params_to_save = copy.deepcopy(params)
#params_to_save.pop("basis", None)
#params_to_save["coefs"] = sol.cells_coefs

#dump_pars(pars_to_save)

n = 20
ts = np.linspace(settings['MODEL']["area_lims"][0, 0], settings['MODEL']["area_lims"][0, 1] - 1e-9, n)
points = [[t] for t in ts]
vals = [[sol.eval(np.array([t]), [0], 1)] for t in ts]

import matplotlib.pyplot as plt
plt.plot(points, vals)
plt.show()

out_dict = {'points':points, 'data':vals}
with open('colloc_solution_I.pkl', 'wb') as out_file:
    pkl.dump(out_dict, out_file)
