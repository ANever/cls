# simple direct problem solving
import matplotlib.pyplot as plt

from solution_global_wip import Solution, eval_dict

from basis import Basis
import itertools
import numpy as np

import yaml

settings_filename = "settings.yaml"

with open(settings_filename, mode="r") as file:
    settings = yaml.safe_load(file)

params = eval_dict(settings["MODEL"], {"np": np})


def popravka(f, g):
    return f * g


def f_collocation_points(N):
    points = np.zeros(N + 1)
    h = 2 / (N + 1)
    points[0] = -1 + h / 2
    for i in range(1, N + 1):
        points[i] = points[i - 1] + h
    return np.array(points).reshape(N + 1, 1)


def prepare_ops():
    pass


def prepare_model(border_weight, colloc_weight, connect_weights):
    # SET ACCURATE PHI
    with open(settings_filename, mode="r") as file:
        settings = yaml.safe_load(file)

    params = eval_dict(settings["MODEL"], {"np": np})

    sol = Solution(**params)

    k1, k2 = connect_weights
    custom_vars = eval_dict(settings["CUSTOM_CONSTS"] | settings["CUSTOM_FUNCS"])
    colloc_ops = list(eval_dict(settings["COLLOC_OPS"], custom_vars).values())
    border_ops = list(eval_dict(settings["BORDER_OPS"], custom_vars).values())
    connect_ops = sol.default_connect_ops([k1, k2])
    # settings = eval_dict(settings)

    power = params["power"]
    c_p_1d = f_collocation_points(power).reshape(power + 1)
    colloc_points = np.array(list(itertools.product(c_p_1d, c_p_1d)))

    connect_points = eval(settings["CONNECT_POINTS"])
    border_points = connect_points

    points = [colloc_points, connect_points, border_points]

    iteration_dict = {
        "points": points,
        "colloc_ops": colloc_ops,
        "border_ops": border_ops,
        "connect_ops": connect_ops,
    }

    sol.cells_coefs *= 0.0
    return sol, iteration_dict


weights = {"border_weight": 1, "colloc_weight": 1, "connect_weights": [1, 1]}
sol, iteration_dict = prepare_model(**weights)
print("start")
for i in range(100):
    old = sol.cells_coefs
    A, b = sol.global_solve(
        # solver="SVD",
        **iteration_dict,
        alpha=1e-3,
        verbose=True,
    )
    speed = 0.3
    sol.cel_coefs = sol.cel_coefs*(1-speed) + old*speed 
    print(i)
    print(np.sum((A @ sol.cells_coefs.ravel() - b) ** 2))
# print(np.linalg.solve(A.T @ A, A.T @ b))
# print(A)
# print(b)
print("plotting")
sol.plot2d()
sol.plot2d(func_num=1)
sol.plot2d(func_num=3)
print("finish")
