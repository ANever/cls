from solution import Solution
from solution import lp as _lp
import copy
from basis import Basis
import itertools
import numpy as np

import yaml

from params import *


border_weight = 10

sol = Solution(**params)
sol.cells_coefs *= 0.0

function_list = ['S', 'I', 'p', 'psi']
variable_list = ['t','x']

customs={'beta': 1,
        'sigma': 0.1,
        'gamma': 1}

def lp(line, function_list=function_list, variable_list = variable_list, customs=customs):
    res = _lp(line, function_list, variable_list)
    print(res)
    return lambda _, u_loc, u_bas, x, x_loc: eval(res, customs | {'u_bas': u_bas, 'u_loc': u_loc, 'x_loc': x_loc})

colloc_left_operators = [lp('- (d/dt) S - beta * ( S * &I ) + sigma /2 * (d/dx)^2 S '),
#lp('- (d/dt) S - beta * ( &S * I + S * &I ) + sigma /2 * (d/dx)^2 S '), #- 1/2/w0 * ( (d/dx)^2 
                lp('(d/dx) I '),
                lp('(d/dx) p - S ')
                ]

colloc_right_operators = [lp('0'),
                #lp('- beta * ( &S * &I ) '), #- 1/2/w0 * ( (d/dx)^2 psi * S + (d/dx) psi * (d/dx) S )', 
                lp('0'),
                lp('0'),
                   ]


border_left_operators = [
    #border conditions
    lambda s, _, u_bas, x, x_loc: int(x[0] > sol.area_lims[0, 0] + small) * int(x[0] < sol.area_lims[0, 1] - small)
    * (u_bas([0, 1], 0))
    * border_weight,
    
    lambda s, _, u_bas, x, x_loc: int(x[0] > sol.area_lims[0, 0] + small) * int(x[0] < sol.area_lims[0, 1] - small)
    * int(x[1] > sol.area_lims[1, 1] - small)
    * ( u_bas([1, 0], 1) - u_bas([0,0],2) * u_bas([0,0],1) )#intS * I)
    * border_weight,
    
    lambda s, _, u_bas, x, x_loc: int(x[0] > sol.area_lims[0, 0] + small) * int(x[0] < sol.area_lims[0, 1] - small)
    * int(x[1] < sol.area_lims[1, 0] + small)
    * (u_bas([0, 0], 2))
    * border_weight,
    #initial conditions
    lambda s, _, u_bas, x, x_loc: int(x[0] < sol.area_lims[0, 0] + small)
    * (u_bas([0, 0], 0))
    * border_weight,
    lambda s, _, u_bas, x, x_loc: int(x[0] < sol.area_lims[0, 0] + small) * int(x[1] < sol.area_lims[1, 0] + small)
    * (u_bas([0, 0], 1))
    * border_weight * 100,
    #terminal conditions
    #lambda s, _, u_bas, x, x_loc: int(x[0] < sol.area_lims[0, 0] + small)
    #* u_bas([0, 0], 3)
    #* border_weight,
]

border_right_operators = [
    #border conditions
    lambda s, u, _, x, x_loc: 0 * border_weight,  # border condition for psi
    lambda s, u, _, x, x_loc: 0 * border_weight,  # border condition for psi
    lambda s, u, _, x, x_loc: 0 * border_weight,  # border condition for psi
    #initial conditions
    lambda s, u, _, x, x_loc: int(x[0] < sol.area_lims[0, 0] + small)
    * (initial_state(x))
    * border_weight,
    lambda s, u, _, x, x_loc: int(x[0] < sol.area_lims[0, 0] + small)
    * 0.1* 100
    * border_weight,
    #terminal conditions
    #lambda s, u, _, x, x_loc: int(x[0] > sol.area_lims[0, 0] + small)
    #* initial_state(x)
    #* border_weight,  # border and initial cond for rho
]

colloc_ops = [colloc_left_operators, colloc_right_operators]
border_ops = [border_left_operators, border_right_operators]


settings_filename = "settings.yaml"


with open(settings_filename, mode="r") as file:
    settings = yaml.safe_load(file)


def f_collocation_points(N):
    points = np.zeros(N + 1)
    h = 2 / (N + 1)
    points[0] = -1 + h / 2
    for i in range(1, N + 1):
        points[i] = points[i - 1] + h
    return np.array(points).reshape(N + 1, 1)
connect_points = eval(settings['CONNECT_POINTS'])
border_points = connect_points


small = 1e-5

c_p_1d = f_collocation_points(power).reshape(power + 1)
colloc_points = np.array(list(itertools.product(c_p_1d, c_p_1d)))


def dir(point: np.array) -> np.array:
    direction = (np.abs(point) == 1) * (np.sign(point))
    return np.array(direction, dtype=int)


points = [colloc_points, connect_points, border_points]

iteration_dict = {
    "points": points,
    "colloc_ops": colloc_ops,
    "border_ops": border_ops,
    #"connect_ops": connect_ops,
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


k = 30
r = np.array((k * sol.cells_coefs.shape))
for j in range(k):
    prev_coefs = copy.deepcopy(sol.cells_coefs)
    prev_eval = sol_eval(sol)
    A, b = sol.global_solve(
        solver="np",
        #svd_threshold=1e-8,
        #return_system=True,
        alpha=1e-4,
        **iteration_dict,
    )
    speed = 0.5
    sol.cells_coefs = (1-speed)*prev_coefs + speed*sol.cells_coefs 
    print(j,' | ', np.max(np.abs(prev_coefs - sol.cells_coefs)),' | ', np.max(np.abs(prev_eval - sol_eval(sol))), ' | ', np.max(np.abs(A @ sol.cells_coefs.ravel() - b)))

sol.plot2d()
sol.plot2d(func_num=1)
sol.plot2d(func_num=2)

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
