import numpy as np

a = 1
b = 1
eps = 0.01

power = 5
params = {
    'n_dims': 2,
    'dim_sizes': np.array([6, 6]),
    'area_lims': np.array([[0,0.1], [0,1]]),
    'power': power,
    'n_funcs': 2,
}

w = 1
k1 = 1
k2 = 1
border_weight = 1
no_need_weight = 0

def initial_state(x):
    t, x = x
    sm_x = 3*x-1
    if (x<2/3 and x>1/3):
        return 12 * (sm_x**2) *(1-sm_x)
    else:
        return 0

def terminal_state(x):
    sm_x = 3*x-1
    if (x<2/3 and x>1/3):
        return 12*sm_x*(1-sm_x)**2
    else:
        return 0