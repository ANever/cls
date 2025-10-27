from solution import Solution
import copy
from basis import Basis
import itertools
import numpy as np
import utils

import yaml

power = 6

function_list = ['S', 'I', 'betaS', 'psi', 'betaSg']
variable_list = ['t','x']

params = {
    "n_dims": len(variable_list),
    "dim_sizes": np.array([6, 6]),
    "area_lims": np.array([[0, 1], [0, 1]]),
    "power": power,
    "n_funcs": len(function_list),
}

def initial_state(x):
    _, x = x
    return 0.9 

border_weight = 50

sol = Solution(**params)
sol.cells_coefs *= 0.01

customs={'beta': 20,
        'sigma': 0.1,
        'gamma': 2,
        'w1': 1.,
        'w2': 20,
        'w3': 1,
        'border_weight': 50,
        'small': 1e-5,
        }

def lp(lines_list, function_list=function_list, variable_list = variable_list, customs=customs):
    line_res = np.array([utils.lp(line, function_list, variable_list, customs=customs) for line in colloc_lines]).T
    return line_res

colloc_lines = [
    '-( (d/dt) S ) + sigma /2 * (d/dx)^2 S - 1/2 * ( (d/dx) S * (d/dx) &psi + (d/dx) &S * (d/dx) psi + S * (d/dx)^2 &psi + &S * (d/dx)^2 psi ) - (d/dt) psi - (1/2 * ( (d/dx) psi * (d/dx) &psi )) = beta*x[1]*( &S * &I ) + 1/2 * ( (d/dx) &S * (d/dx) &psi + &S * (d/dx)^2 &psi ) ',
    '(d/dt) I - &betaSg * I + gamma * I = w1*( &I )**2 + w2*x[1] * (0.5 + &I ) + w3*(1- x[1])**2',
    '(d/dx) I = 0',
    '(d/dx) betaSg = 0',
    '(d/dx) betaS = 0'
]

borders_string = 'int(x[0] > sol.area_lims[0, 0] + small) * int(x[0] < sol.area_lims[0, 1] - small)'
initial_string = 'int(x[0] < sol.area_lims[0, 0] + small)'
terminal_string = 'int(x[0] > sol.area_lims[0, 1] - small)'
border_lines = [
    borders_string + '* (d/dx) S * border_weight = 0 * border_weight',
    borders_string + '* (d/dx) I * border_weight = 0 * border_weight',
    borders_string + '* (d/dx) psi * border_weight = 0 * border_weight',
    initial_string + '* (d/dx) S * border_weight = 0.9 * border_weight',
    initial_string + '* (d/dx) I * border_weight = 0.1 * border_weight',
    terminal_string + '* (d/dx) psi * border_weight = 0 * border_weight',
    'int(x[1] > sol.area_lims[1, 1] - small) * betaSg * border_weight = betaS * border_weight ',
    'int(x[1] < sol.area_lims[1, 0] + small)* betaS* border_weight = 0',
    ]

colloc_ops = lp(colloc_lines, customs=customs)
border_ops = lp(border_lines, customs=customs)

'''
settings_filename = "settings.yaml"


with open(settings_filename, mode="r") as file:
    settings = yaml.safe_load(file)
'''

dots = np.linspace(-0.9,0.9,power)
connect_points = np.array([[-1, i] for i in dots] + [[1, i] for i in dots]+
                [[i,-1] for i in dots] + [[i,1] for i in dots])
border_points = connect_points



small = 1e-5

c_p_1d = utils.f_collocation_points(power+1).reshape(power + 2)
colloc_points = np.array(list(itertools.product(c_p_1d, c_p_1d)))

points = [colloc_points, connect_points, border_points]

iteration_dict = {
    "points": points,
    "colloc_ops": colloc_ops,
    "border_ops": border_ops,
}

import copy

n = 20
ts = np.linspace(params["area_lims"][0, 0], params["area_lims"][0, 1] - small, n)
xs = np.linspace(params["area_lims"][1, 0], params["area_lims"][1, 1] - small, n)


def sol_eval(sol, ts=ts, xs=xs):
    res_array = np.zeros((len(ts), len(xs)))
    for i in range(len(ts)):
        t = ts[i]
        for j in range(len(xs)):
            x = xs[j]
            res_array[i, j] = sol.eval(np.array([t, x]), [0, 0])

    return res_array


k = 200
r = np.array((k * sol.cells_coefs.shape))
for j in range(k):
    prev_coefs = copy.deepcopy(sol.cells_coefs)
    prev_eval = sol_eval(sol)
    A, b = sol.global_solve(
        solver="np",
        alpha=1e-8,
        **iteration_dict,
    )
    speed = (0.2 + (1/2/(j+1)))/2
    sol.cells_coefs = (1-speed)*prev_coefs + speed*sol.cells_coefs 
    sol_change = np.max(np.abs(prev_eval - sol_eval(sol)))
    print(j,' | ', np.max(np.abs(prev_coefs - sol.cells_coefs)),' | ', sol_change , ' | ', np.max(np.abs(A @
sol.cells_coefs.ravel() - b)))
    if sol_change < 1e-5:
        break

for i in range(len(function_list)):
    utils.plot2d(sol, func_num=i)

params_to_save = copy.deepcopy(params)
params_to_save.pop("basis", None)
params_to_save["coefs"] = sol.cells_coefs

import yaml
import pickle as pkl

points = []
vals = []
for t in ts:
    for x in xs:
        points.append([t,x])
        vals.append([sol.eval(np.array([t, x]), [0, 0])])

out_dict = {'points':points, 'data':vals}
with open('colloc_solution_S.pkl', 'wb') as out_file:
    pkl.dump(out_dict, out_file)
