from solution import Solution
from utils import _lp, plot, eval_dict
import copy
from basis import Basis
import itertools
import numpy as np
import copy

import yaml
import pickle as pkl


settings_filename = "settings/simplest_mfg.yaml"

with open(settings_filename, mode="r") as file:
    settings = yaml.safe_load(file)

def lp(line, function_list=settings['OUT_VAR_NAMES'], variable_list =settings['IN_VAR_NAMES'], customs=customs):
        res = _lp(line, function_list=function_list, variable_list=variable_list, customs=customs)
        print('res', res)
        return res

def prepare_settings(settings):
    settings['MODEL']['n_dims'] = len(settings['IN_VAR_NAMES'])
    settings['MODEL']['n_funcs'] = len(settings['OUT_VAR_NAMES'])

    customs = settings['CUSTOMS']
    for key in customs.keys():
        if isinstance(customs[key], str):
            compile(customs[key], '<string>', 'eval')
            customs[key] = eval(customs[key], locals() | customs)

    for cond in ['COLLOC_OPS', 'BORDER_OPS']:
        for side in ['left', 'right']:
            for i, line in enumerate(settings[cond][side]):
                settings[cond][side][i] = lp(line)

    colloc_ops = list(settings['COLLOC_OPS'].values())
    border_ops = list(settings['BORDER_OPS'].values())

    connect_points = np.array(eval(settings['CONNECT_POINTS']))
    border_points = connect_points

    power = settings['MODEL']['power']
    colloc_points = np.reshape(np.linspace(-1,1,power+2), (power+2,1))
    points = [colloc_points, connect_points, border_points]

    iteration_dict = {
        "points": points,
        "colloc_ops": colloc_ops,
        "border_ops": border_ops,
        #"connect_ops": connect_ops,
    }

    return settings, iteration_dict

settings, iteration_dict = prepare_settings(settings)
sol = Solution(**eval_dict(settings['MODEL'], {'np':np}))
sol.cells_coefs *= 0.0

n = 20
ts = np.linspace(settings['MODEL']["area_lims"][0, 0], settings['MODEL']["area_lims"][0, 1] - 0.00001, n)

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

k = 50
#r = np.array((k * sol.cells_coefs.shape))
for j in range(k):
    prev_coefs = copy.deepcopy(sol.cells_coefs)
    #prev_eval = sol_eval(sol)
    A, b = sol.global_solve(
        solver="np",
        #svd_threshold=1e-8,
        #return_system=True,
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

import json
from json import JSONEncoder
import numpy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


with open("data.json", "w") as f:
    json.dump(params_to_save, f, cls=NumpyArrayEncoder)

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
