from solution import Solution
from solution import lp as _lp
import copy
from basis import Basis
import itertools
import numpy as np

import yaml

power = 5   


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
        'w3': 1}

def lp(line, function_list=function_list, variable_list = variable_list, customs=customs):
    res = _lp(line, function_list, variable_list)
    print(res)
    return lambda _, u_loc, u_bas, x, x_loc: eval(res, customs | {'u_bas': u_bas, 'u_loc': u_loc, 'x_loc': x_loc, 'x':x})

colloc_left_operators = [lp('-( (d/dt) S ) + sigma /2 * (d/dx)^2 S - 1/2 * ( (d/dx) S * (d/dx) &psi + (d/dx) &S * (d/dx) psi + S * (d/dx)^2 &psi + &S * (d/dx)^2 psi ) '),
lp('- (d/dt) psi - (1/2 * ( (d/dx) psi * (d/dx) &psi ))'),
                lp('(d/dt) I - &betaSg * I + gamma * I'),
                lp('(d/dx) I'),
                lp('(d/dx) betaSg'),
                lp('(d/dx) betaS')
                ]

colloc_right_operators = [lp('beta*x[1]*( &S * &I ) + 1/2 * ( (d/dx) &S * (d/dx) &psi + &S * (d/dx)^2 &psi ) '),
                lp('w1*( &I )**2 + w2*x[1] * (0.5 + &I ) + w3*(1- x[1])**2'),
                lp('0'),
                lp('0'),
                lp('0'),
                lp('&S * beta * x[1]'),
                   ]

border_left_operators = [
    #border conditions
    lambda s, _, u_bas, x, x_loc: int(x[0] > sol.area_lims[0, 0] + small) * int(x[0] < sol.area_lims[0, 1] - small)
    * (u_bas([0, 1], 0))
    * border_weight,
    lambda s, _, u_bas, x, x_loc: int(x[0] > sol.area_lims[0, 0] + small) * int(x[0] < sol.area_lims[0, 1] - small)
    * (u_bas([0, 1], 1))
    * border_weight,
    
    lambda s, _, u_bas, x, x_loc: int(x[0] > sol.area_lims[0, 0] + small) * int(x[0] < sol.area_lims[0, 1] - small)
    * (u_bas([0, 1], 3))
    * border_weight,
    
    #lambda s, u_loc, u_bas, x, x_loc: 1*#int(x[0] > sol.area_lims[0, 0] + small) * int(x[0] < sol.area_lims[0, 1] - small)*
    #int(x[1] > sol.area_lims[1, 1] - small)
    #* ( u_bas([1, 0], 1) - u_loc([0,0],2) * u_bas([0,0],1) + 2 * u_bas([0,0],1) )#intS * I)
    #* border_weight,
    
    lambda s, u_loc, u_bas, x, x_loc: 1*#int(x[0] > sol.area_lims[0, 0] + small) * int(x[0] < sol.area_lims[0, 1] - small)*
    int(x[1] > sol.area_lims[1, 1] - small)
    * u_bas([0, 0], 4) #intS * I)
    * border_weight,
    
    lambda s, _, u_bas, x, x_loc: 1* #int(x[0] > sol.area_lims[0, 0] + small) * int(x[0] < sol.area_lims[0, 1] - small)
    int(x[1] < sol.area_lims[1, 0] + small)
    * (u_bas([0, 0], 2))
    * border_weight,
    #initial conditions
    lambda s, _, u_bas, x, x_loc: int(x[0] < sol.area_lims[0, 0] + small)
    * (u_bas([0, 0], 0))
    * border_weight*100,
    lambda s, _, u_bas, x, x_loc: int(x[0] < sol.area_lims[0, 0] + small)
    * (u_bas([0, 0], 1))
    * border_weight,
    #terminal conditions
    lambda s, _, u_bas, x, x_loc: int(x[0] > sol.area_lims[0, 1] - small)
    * u_bas([0, 0], 3)
    * border_weight,
]

border_right_operators = [
    #border conditions
    lambda s, u, _, x, x_loc: 0 * border_weight,  # border condition for psi
    lambda s, u, _, x, x_loc: 0 * border_weight,  # border condition for psi
    lambda s, u, _, x, x_loc: 0 * border_weight,  # border condition for psi
    lambda s, u, _, x, x_loc: u([0,0],2) * border_weight,  # border condition for psi
    lambda s, u, _, x, x_loc: 0 * border_weight,  # border condition for psi
    #initial conditions
    lambda s, u, _, x, x_loc: int(x[0] < sol.area_lims[0, 0] + small)
    * (initial_state(x))
    * border_weight*100,
    lambda s, u, _, x, x_loc: int(x[0] < sol.area_lims[0, 0] + small)
    * 0.1
    * border_weight,
    #terminal conditions
    lambda s, u, _, x, x_loc: 0 * border_weight,  # border condition for psi
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
#connect_points = eval(settings['CONNECT_POINTS'])
#border_points = connect_points

dots = np.linspace(-0.9,0.9,power)
connect_points = np.array([[-1, i] for i in dots] + [[1, i] for i in dots]+
                [[i,-1] for i in dots] + [[i,1] for i in dots])
border_points = connect_points



small = 1e-5

c_p_1d = f_collocation_points(power+1).reshape(power + 2)
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


k = 400
r = np.array((k * sol.cells_coefs.shape))
for j in range(k):
    prev_coefs = copy.deepcopy(sol.cells_coefs)
    prev_eval = sol_eval(sol)
    A, b = sol.global_solve(
        solver="np",
        #svd_threshold=1e-8,
        #return_system=True,
        alpha=1e-8,
        **iteration_dict,
    )
    speed = (0.05 + (1/2/(j+1)))/2
    sol.cells_coefs = (1-speed)*prev_coefs + speed*sol.cells_coefs 
    sol_change = np.max(np.abs(prev_eval - sol_eval(sol)))
    print(j,' | ', np.max(np.abs(prev_coefs - sol.cells_coefs)),' | ', sol_change , ' | ', np.max(np.abs(A @
sol.cells_coefs.ravel() - b)))
    if sol_change < 1e-5:
        break

#for i in range(len(function_list)):
#    sol.plot2d(func_num=i)

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
