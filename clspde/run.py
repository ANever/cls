import matplotlib.pyplot as plt
from solution import Solution
from basis import Basis
import itertools
import numpy as np

from .params import *

params['basis']=Basis(params['power'])
sol = Solution(**params)
sol.cells_coefs *= 0.0

def f_collocation_points(N):
    points = np.zeros(N+1)
    h = 2/(N+1)
    points[0] = -1 + h/2
    for i in range(1, N+1):
        points[i] = points[i-1] + h
    return np.array(points).reshape(N+1,1)

c_p_1d = f_collocation_points(power).reshape(power+1)

colloc_points = np.array(list(itertools.product(c_p_1d, c_p_1d)))

connect_points = np.array([[-1, 0.5], [1, 0.5],
                            [0.5, -1], [0.5, 1],
                            [-1, -0.5], [1, -0.5],
                            [-0.5, -1], [-0.5, 1],])
                            
border_points = connect_points

colloc_left_operators = [lambda u_loc, u_bas, x, x_loc:  (u_bas([1,0],0)-eps*u_bas([0,2],0)
                                                                                            -(u_bas([0,1],0)*u_loc([0,1],1)+
                                                                                              u_loc([0,1],0)*u_bas([0,1],1)+

                                                                                             u_loc([0,0],0)*u_bas([0,2],1)+
                                                                                             u_bas([0,0],0)*u_loc([0,2],1))
                                                                                          ) * w**2,

                        lambda u_loc, u_bas, x, x_loc:  (-u_bas([1,0],1)-eps*u_bas([0,2],1)
                                                                                             +(u_bas([0,1],1)*u_loc([0,1],1))
                                                                                            ) * w**2 ,
                                                                                            ]

def popravka(f,g):
    return f*g
colloc_right_operators = [lambda u_loc, u_nei, x, x_loc: (-popravka(u_loc([0,1],0), u_loc([0,1],1)) - popravka(u_loc([0,0],0), u_loc([0,2],1))
                                                         ) * w**2,
                          lambda u_loc, u_nei, x, x_loc: (popravka(u_loc([0,0],0),  u_loc([0,0],1))) * w**2,
                          ]
colloc_ops = [colloc_left_operators, colloc_right_operators]

small = 1e-5

border_left_operators = [
    lambda _, u_bas, x, x_loc: int(x[0]>sol.area_lims[0,0]+small)*int(x[0]<sol.area_lims[0,1]-small)*(u_bas([0,0],1)) * border_weight,
    lambda _, u_bas, x, x_loc: int(x[0]<sol.area_lims[0,0]+small)*(u_bas([0,0],1)) * border_weight,
    lambda _, u_bas, x, x_loc: int(x[0]>sol.area_lims[0,0]+small)*u_bas([0,0],0) * border_weight,
    ]

border_right_operators = [
    lambda u, _, x, x_loc: 0 * border_weight, # border condition for psi
    lambda u, _, x, x_loc: int(x[0]<sol.area_lims[0,0]+small) * (-terminal_state(x[1])) * border_weight, 
    lambda u, _, x, x_loc: int(x[0]>sol.area_lims[0,0]+small) * initial_state(x) * border_weight, # border and initial cond for rho
]

border_ops = [border_left_operators, border_right_operators]

def f_collocation_points(N):
    points = np.zeros(N+1)
    h = 2/(N+1)
    points[0] = -1 + h/2
    for i in range(1, N+1):
        points[i] = points[i-1] + h
    return np.array(points).reshape(N+1,1)

c_p_1d = f_collocation_points(power).reshape(power+1)
colloc_points = np.array(list(itertools.product(c_p_1d, c_p_1d)))
connect_points = np.array([[-1, 0.5], [1, 0.5],
                            [0.5, -1], [0.5, 1],
                            [-1, -0.5], [1, -0.5],
                            [-0.5, -1], [-0.5, 1],
                            [-1, 0], [1, 0],
                            [0, -1], [0, 1],
                            ])
border_points = connect_points
points=[colloc_points, connect_points ,border_points]

connect_left_operators = []
connect_right_operators = []

def dir(point: np.array) -> np.array:
    direction = (np.abs(point) == 1) * (np.sign(point))
    return np.array(direction, dtype=int)
for func_num in range(sol.n_funcs):
    connect_left_operators += [lambda _, u_bas, x, x_loc, func_num=func_num: k1*u_bas(0*dir(x_loc),func_num) + k2*np.sum(dir(x_loc))*u_bas([0,1],func_num)]
    connect_right_operators += [lambda _, u_bas, x, x_loc, func_num=func_num: k1*u_bas(0*dir(x_loc),func_num) -  k2*np.sum(dir(x_loc))*u_bas([0,1],func_num)]
connect_ops = [connect_left_operators, connect_right_operators]

iteration_dict = {'points':points,
        'colloc_ops':colloc_ops,
        'border_ops':border_ops,
       'connect_ops':connect_ops
}

import copy

n=20
ts = np.linspace(params['area_lims'][0,0],params['area_lims'][0,1] - small, n)
xs = np.linspace(params['area_lims'][1,0],params['area_lims'][1,1] - small, n)

def eval(sol, ts=ts, xs=xs):
    res_array = np.zeros((len(ts), len(xs)))
    for i in range(len(ts)):
        t = ts[i]
        for j in range(len(xs)):
            x = xs[j]
            res_array[i,j] = sol.eval(np.array([t,x]), [0,0])
            
    return res_array

k=10
r = np.array((k *sol.cells_coefs.shape))
for j in range(k):
    prev_eval = eval(sol)
    prev_coefs = copy.deepcopy(sol.cells_coefs)
    A, b = sol.global_solve(solver='np', svd_threshold=1e-8, return_system=True, alpha = 0.00, **iteration_dict)
    # print(j,' | ', np.max(np.abs(prev_coefs - sol.cells_coefs)),' | ', np.max(np.abs(prev_eval - eval(sol))))

params_to_save = copy.deepcopy(params)
params_to_save.pop('basis', None)
params_to_save['coefs'] = sol.cells_coefs

import json
from json import JSONEncoder
import numpy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

with open('data.json', 'w') as f:
    json.dump(params_to_save, f, cls=NumpyArrayEncoder)