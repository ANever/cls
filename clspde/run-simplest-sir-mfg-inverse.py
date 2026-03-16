from solution import Solution
from utils import _lp, plot
import copy
from basis import Basis
import itertools
import numpy as np

import yaml

power = 6


function_list = ['S', 'I', 'uS', 'uI','beta_max']
variable_list = ['t']

params = {
    "n_dims": len(variable_list),
    "dim_sizes": np.array([6]),
    "area_lims": np.array([[-1, 1]]),
    "power": power,
    "n_funcs": len(function_list),
}

border_weight = 10

sol = Solution(**params)
sol.cells_coefs *= 0.0


customs={#'beta_max_loc': 'lambda u_: u_(func=4)', #20,
        'cost_inf': 4.,
        'gamma': 2,
        'S0':0.7,
        'c': 'lambda u_: np.abs(u_(func=4) *( u_(func=3) - u_(func=2) ) *u_(func=1) / 2.)',
        'approximation': 'lambda u_: -6.63320111e-01 + np.sqrt(c(u_)) *8.92654544e-01 - c(u_) * 6.99385982e-02 + c(u_)**2* 2.97831057e-04 - c(u_)**3 * 1.08549568e-06 + c(u_)**4*1.68015701e-09 +np.exp(-c(u_))*  3.66041065e-02 + np.exp(-2*c(u_))*  4.74148560e-01',
        'c_sign': 'lambda u_: np.sign( - ( u_(func=3) - u_(func=2) ))',
        'alpha': 'lambda u_: c_sign(u_) * approximation(u_)',
        'beta': 'lambda u_: 1/(1+np.exp(-alpha(u_)))',
        }

for key in customs.keys():
    if isinstance(customs[key], str):
        compile(customs[key], '<string>', 'eval')
        customs[key] = eval(customs[key], locals() | customs)

print(customs)
S0 = 0.7
def lp(line, function_list=function_list, variable_list = variable_list, customs=customs):
    res = _lp(line, function_list=function_list, variable_list=variable_list, customs=customs)
    print('res', res)
    return res #lambda _, u_loc, u_bas, x, x_loc: eval(res, customs | {'u_bas': u_bas, 'u_loc': u_loc, 'x_loc': x_loc})

colloc_left_operators = [lp('- (d/dt) S - ( &beta_max * beta(u_loc) * ( S * &I ) + beta_max * beta(u_loc) * ( &S * &I )) '),
                lp('- (d/dt) I + ( &beta_max * beta(u_loc) * ( S * &I ) + beta_max * beta(u_loc) * ( &S * &I )) - gamma * I '),
                lp('- (d/dt) uS - ( beta(u_loc) * I * ( - uS + uI ))'),
                lp('- (d/dt) uI - ( -gamma * uI )'),
                lp('1000 * (d/dt) beta_max'),
                ]

colloc_right_operators = [lp('-( &beta_max * beta(u_loc) * ( &S * &I ))'),
                #lp('- beta * ( &S * &I ) '),
                lp('( &beta_max * beta(u_loc) * ( &S * &I ))'),
                lp('alpha(u_loc)**2'),
                lp('cost_inf'),
                lp('0'),
                   ]


border_left_operators = [
    #initial conditions
    lambda s, _, u_bas, x, x_loc: int(x[0] < sol.area_lims[0, 0] + small) * (u_bas([0], 0))
    * border_weight,
    lambda s, _, u_bas, x, x_loc: int(x[0] < sol.area_lims[0, 0] + small) * (u_bas([0], 1))
    * border_weight,
    #terminal conditions
    lambda s, _, u_bas, x, x_loc: int(x[0] > sol.area_lims[0, 1] - small) * (u_bas([0], 2))
    * border_weight,
    lambda s, _, u_bas, x, x_loc: int(x[0] > sol.area_lims[0, 1] - small) * (u_bas([0], 3))
    * border_weight,
]

border_right_operators = [
    lambda s, u, _, x, x_loc: int(x[0] < sol.area_lims[0, 0] + small)
    * ( S0 )
    * border_weight,
    lambda s, u, _, x, x_loc: int(x[0] < sol.area_lims[0, 0] + small)
    * ( 1 - S0 )
    * border_weight,
    lambda s, u, _, x, x_loc: 0 * border_weight,
    lambda s, u, _, x, x_loc: 0 * border_weight,
]

custom_points_left_operators = [
    ]
custom_points_left_right = [
    ]

colloc_ops = [colloc_left_operators, colloc_right_operators]
border_ops = [border_left_operators, border_right_operators]

'''
settings_filename = "settings.yaml"


with open(settings_filename, mode="r") as file:
    settings = yaml.safe_load(file)
'''

connect_points = np.array([[-1], [1]])
border_points = connect_points

small = 1e-5
colloc_points = np.reshape(np.linspace(-1,1,power+2), (power+2,1))

def dir(point: np.array) -> np.array:
    direction = (np.abs(point) == 1) * (np.sign(point))
    return np.array(direction, dtype=int)

points = [colloc_points, connect_points, border_points]

iteration_dict = {
    "points": points,
    "colloc_ops": colloc_ops,
    "border_ops": border_ops,
}

import copy

n = 20
ts = np.linspace(params["area_lims"][0, 0], params["area_lims"][0, 1] - small, n)

k = 50
r = np.array((k * sol.cells_coefs.shape))
for j in range(k):
    prev_coefs = copy.deepcopy(sol.cells_coefs)
    #prev_eval = sol_eval(sol)
    A, b = sol.global_solve(
        solver="np",
        #svd_threshold=1e-8,
        #return_system=True,
        #alpha=1e-4,
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
    
    
import yaml
import pickle as pkl

points = []
vals = []
for t in ts:
    points.append([t])
    vals.append([sol.eval(np.array([t]), [0], 1)])

import matplotlib.pyplot as plt
plt.plot(points, vals)
plt.show()

out_dict = {'points':points, 'data':vals}
#with open('colloc_solution_I.pkl', 'wb') as out_file:
#    pkl.dump(out_dict, out_file)
