from clspde.utils import plot, eval_dict
from clspde.prepare import from_file, prepare_settings
from clspde.solution import Solution
import itertools
import numpy as np
import copy
import yaml
import pickle as pkl

settings_filename = "settings/simplest_mfg.yaml"
settings, sol_mes, iteration_dict = from_file(settings_filename)
with open('colloc_solution_coefs.pkl', 'rb') as in_file:
    coefs = pkl.load(in_file)

print(coefs['coefs'].shape)

sol_mes.cells_coefs = coefs['coefs']



settings_filename = "settings/simplest_mfg_inverse.yaml"
with open(settings_filename, mode="r") as file:
    settings = yaml.safe_load(file)



settings['CUSTOMS']['I_info'] = lambda x : sol_mes.eval(point=x, der=[0], func=1, cells_closed_right=True)
settings, iteration_dict = prepare_settings(settings)
sol = Solution(**eval_dict(settings['MODEL'], {'np':np}))
sol.cells_coefs *= 0.0


print('BETA', (sol_mes.eval([0.0001],[1],func=1)/0.3 + 2)/0.7, sol_mes.eval([0.0001],[1],func=1))

n = 20
ts = np.linspace(settings['MODEL']["area_lims"][0, 0], settings['MODEL']["area_lims"][0, 1] - 1e-9, n)
def eval_error():
    er = [0]*5
    for func in range(4):
        for t in ts:
            er[func] += (sol.eval([t],[0],func) - sol_mes.eval([t],[0],func))**2
        er[4] = sol.eval([0],[0],func=4)
    return er


k = 50
for j in range(k):
    prev_coefs = copy.deepcopy(sol.cells_coefs)
    #prev_eval = sol_eval(sol)
    A, b = sol.global_solve(
        solver="np",
        #svd_threshold=1e-8,
        alpha=0,#1e-4,
        **iteration_dict,
    )
    speed = 0.8
    sol.cells_coefs = (1-speed)*prev_coefs + speed*sol.cells_coefs 
    print(j,' | ', np.max(np.abs(prev_coefs - sol.cells_coefs)),' | ', eval_error())
plot(sol)

#params_to_save = copy.deepcopy(params)
#params_to_save.pop("basis", None)
#params_to_save["coefs"] = sol.cells_coefs

dump_pars(pars_to_save)

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
