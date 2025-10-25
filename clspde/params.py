import numpy as np

power = 5
params = {
    "n_dims": 2,
    "dim_sizes": np.array([6, 6]),
    "area_lims": np.array([[0, 1], [0, 1]]),
    "power": power,
    "n_funcs": 3,
}
def initial_state(x):
    _, x = x
    sm_x = 3 * x - 1
    if x < 2 / 3 and x > 1 / 3:
        return 12 * (sm_x**2) * (1 - sm_x)
    else:
        return 0

