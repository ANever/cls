
#simple direct problem solving 

import matplotlib.pyplot as plt

from .solution_gf_wip import Solution
from .basis import Basis
import itertools
import numpy as np

# SET ACCURATE PHI

a = 1
b = 1

eps = 0.05

power = 6
params = {
    'n_dims': 2,
    'dim_sizes': np.array([6, 6]),
    'area_lims': np.array([[0,0.1], [0,1]]),
    'power': power,
    'basis': Basis(power),
    'n_funcs': 2,
}
sol = Solution(**params)
w = (sol.steps[0]/2)

def f_collocation_points(N):
    points = np.zeros(N+1)
    h = 2/(N+1)
    points[0] = -1 + h/2
    for i in range(1, N+1):
        points[i] = points[i-1] + h
    return np.array(points).reshape(N+1,1)

# c_p_1d = f_collocation_points(int(np.ceil(power/2))).reshape(int(np.ceil(power/2))+1)
# power = int(3/2*power)
c_p_1d = f_collocation_points(power).reshape(power+1)

colloc_points = np.array(list(itertools.product(c_p_1d, c_p_1d)))


connect_points = np.array([[-1, 0.5], [1, 0.5],
                            [0.5, -1], [0.5, 1],
                            [-1, -0.5], [1, -0.5],
                            [-0.5, -1], [-0.5, 1],])
                            
border_points = connect_points
# border_points = np.array([[-1, 0.5], [1, 0.5],
#                             [0.5, -1], [0.5, 1],
#                             [-1, -0.5], [1, -0.5],
#                             [-0.5, -1], [-0.5, 1],])

colloc_left_operators = [lambda u_loc, u_bas, x, x_loc:  (u_bas([1,0],0)-eps*u_bas([0,2],0)
                                                                                            # '''
                                                                                            # +(2*u_loc([0,1],1)*u_bas([0,1],0)*u_loc([0,0],0)
                                                                                            # +u_bas([0,2],1)*u_loc([0,1],0)**2)/2
                                                                                            # '''

                                                                                                # +(2*(u_bas([0,1],1)*u_loc([0,1],0)*u_loc([0,0],0)+
                                                                                                # +u_loc([0,1],1)*u_bas([0,1],0)*u_loc([0,0],0)+
                                                                                                # +u_loc([0,1],1)*u_loc([0,1],0)*u_bas([0,0],0))+

                                                         
                                                                                                # +u_bas([0,2],1)*u_loc([0,1],0)**2+
                                                                                                # +2*u_loc([0,2],1)*u_bas([0,1],0)*u_loc([0,1],0))/2/3 
                                                                                            
                                                                                            #new_control
                                                                                            -(u_bas([0,1],0)*u_loc([0,1],1)+
                                                                                              u_loc([0,1],0)*u_bas([0,1],1)+

                                                                                             u_loc([0,0],0)*u_bas([0,2],1)+
                                                                                             u_bas([0,0],0)*u_loc([0,2],1))/2

                                                                                            # -(u_bas([0,1],0)*u_loc([0,1],1)+
                                                                                            #   u_bas([0,0],0)*u_loc([0,2],1))

                                                                                          ) * w**2,
                        lambda u_loc, u_bas, x, x_loc:  (-u_bas([1,0],1)-eps*u_bas([0,2],1)
                                                                                            # ТАк НЕльзя
                                                                                            # +(u_loc([0,0],0)*u_bas([0,1],1)**2)/2
                                                                                            
                                                                                            # +(u_bas([0,0],0)*u_loc([0,1],1)**2)/2

                                                                                            #'''
                                                                                            # -(u_bas([0,0],0)*u_loc([0,1],1)**2+
                                                                                            # 2*u_loc([0,0],0)*u_bas([0,1],1)*u_loc([0,1],1))/2/3

                                                                                            #Tailor linearisation (!!!mind right side)
                                                                                            #'''
                                                                                            # -(u_bas([0,0],0)*u_loc([0,1],1)**2+
                                                                                            # 2*u_loc([0,0],0)*u_bas([0,1],1)*u_loc([0,1],1))/2


                                                                                            #new_control
                                                                                            +(u_bas([0,1],1)*u_loc([0,1],1))
                                                                                            ) * w**2 ,]
colloc_right_operators = [lambda u_loc, u_nei, x, x_loc: 0, #2*u_loc([0,0],0)*u_loc([0,1],1)**2 * w**2, #0,                                                        
                          lambda u_loc, u_nei, x, x_loc: 0]
colloc_ops = [colloc_left_operators, colloc_right_operators]

# def p(x):
#     return 6*x*(1-x)

# def initial_state(point):
#     t, x = point
#     if t == 0:
#         return 1
#     else:
#         return 10*(0.1 - t)



# def terminal_state(x):
#     return 12*x*(1-x)*(1-x)

# def initial_state(point):
#     t, x = point
#     return 12*x*x*(1-x)

# def initial_state(point):
#     t, x = point
#     # return np.array([0.98, 0.02, 0]) * np.exp((-((x-0.5)/sigma)**2)/2)#*(1/sigma/np.sqrt(2*3.141592)) #12*x*x*(1-x)
#     sm_x = 3*x-1
#     if (x<2/3 and x>1/3):
#         return 12*sm_x**2 * (1-sm_x) *3
#     else:
#         return 0


def initial_state(x):
    t, x = x
    sm_x = 3*x-1
    if (x<2/3 and x>1/3):
        return 12*sm_x**2 *(1-sm_x) *3
    else:
        return 0


def terminal_state(x):
    sm_x = 3*x-1
    if (x<2/3 and x>1/3):
        return 12*sm_x*(1-sm_x)**2 *3
    else:
        return 0

border_weight = 10
small = 1e-5

border_left_operators = [#lambda _, u_bas, x, x_loc: int(x[0]>sol.area_lims[0,0]+small)*u_bas([0,0],1) * border_weight,
                         lambda _, u_bas, x, x_loc: int(x[0]>sol.area_lims[0,0]+small)*int(x[0]<sol.area_lims[0,1]-small)*(u_bas([0,0],1)) * border_weight,
                         
                        #  lambda _, u_bas, x, x_loc: int(x[0]>sol.area_lims[0,1]-small)*(u_bas([0,0],1)-u_bas([0,0],0)) * border_weight,
                         lambda _, u_bas, x, x_loc: int(x[0]>sol.area_lims[0,1]-small)*(u_bas([0,0],1)-u_bas([0,0],0)) * border_weight,
                         
                         lambda _, u_bas, x, x_loc: int(x[0]<sol.area_lims[0,1]-small)*u_bas([0,0],0) * border_weight,
                            ]

border_right_operators = [#lambda u, _, x, x_loc: int(x[0]>sol.area_lims[0,1]-small) * (u([0,0],0)-p(x[1])) * border_weight,
                          lambda u, _, x, x_loc: 0 * border_weight, # border condition for psi
                        
                        #   lambda u, _, x, x_loc: int(x[0]>sol.area_lims[0,1]-small) * (-terminal_state(x[1])) * border_weight, # terminal condition for psi
                          lambda u, _, x, x_loc: int(x[0]>sol.area_lims[0,1]-small) * (-terminal_state(x[1])) * border_weight, # terminal condition for psi
                        
                          lambda u, _, x, x_loc: int(x[0]<sol.area_lims[0,1]-small)*initial_state(x) * border_weight, # border and initial cond for u
                            ]
border_ops = [border_left_operators, border_right_operators]

def f_collocation_points(N):
    points = np.zeros(N+1)
    h = 2/(N+1)
    points[0] = -1 + h/2
    for i in range(1, N+1):
        points[i] = points[i-1] + h
    return np.array(points).reshape(N+1,1)

# c_p_1d = f_collocation_points(int(np.ceil(power/2))).reshape(int(np.ceil(power/2))+1)

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
    # connect_left_operators += [lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) + np.sum(dir(x_loc))*int(abs(x_loc[1])==1)*u_bas([0,1],func_num)]
    # connect_right_operators += [lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) -  np.sum(dir(x_loc))*int(abs(x_loc[1])==1)*u_bas([0,1],func_num)]
    connect_left_operators += [lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) + np.sum(dir(x_loc))*u_bas([0,1],func_num)]
    connect_right_operators += [lambda _, u_bas, x, x_loc, func_num=func_num: u_bas(0*dir(x_loc),func_num) -  np.sum(dir(x_loc))*u_bas([0,1],func_num)]
connect_ops = [connect_left_operators, connect_right_operators]

iteration_dict = {'points':points,
        'colloc_ops':colloc_ops,
        'border_ops':border_ops,
       'connect_ops':connect_ops
}

sol.cells_coefs *= 0.0

sol.precalculate_basis(np.concatenate([colloc_points, connect_points]), 2)

for i in range(3):
    print(i)
    sol.global_solve(solver='np',svd_threshold=1e-8, **iteration_dict)