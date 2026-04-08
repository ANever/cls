import copy
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from scipy.special import roots_legendre
import numbers
import yaml

import json
from json import JSONEncoder
import numpy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def dump_pars(params_to_save):
    with open("data.json", "w") as f:
        json.dump(params_to_save, f, cls=NumpyArrayEncoder)


def eval_dict(d, kwargs={}, recursion=0):
    if recursion == 0:
        for key in d.keys():
            if key not in ["eq_string", "act", "right_side"]:
                if not isinstance(d[key], numbers.Number):
                    d[key] = eval(str(d[key]), kwargs)
        return d
    else:
        for key in d.keys():
            if key not in ["eq_string", "act", "right_side"]:
                d[key] = eval_dict(d[key], kwargs | d, recursion - 1)
        return d


def concat(args):
    if len(args) == 2:
        return _concat(args[0], args[1])
    else:
        return _concat(args[0], concat(args[1:]))
        
def _concat(a: np.array, b: np.array):
    a = np.array(a)
    b = np.array(b)
    if b.size == 0:
        return a
    if a.size == 0:
        return a
    else:
        return np.concatenate((a, b))


def f_collocation_points(N):
    points = roots_legendre(N+1)[0]
    return np.array(points).reshape(N + 1, 1)


def dir(point: np.array) -> np.array:
    direction = (np.abs(point) == 1) * (np.sign(point))
    return np.array(direction, dtype=int)


def plot(solution, n=100):
    func = np.zeros(n)
    grid = np.linspace(
        solution.area_lims[0, 0], solution.area_lims[0, 1], n, endpoint=False
    )
    for f in range(solution.n_funcs):
        for i in range(len(grid)):
            func[i] = solution.eval(grid[i], [0], f)
        plt.plot(func)
        plt.show()

def plot2d(solution, n=100, x_lims=None, y_lims=None, func_num=0, derivatives=[0, 0]):
    func = np.zeros((n, n))
    if x_lims is None:
        x_lims = solution.area_lims[0]
    if y_lims is None:
        y_lims = solution.area_lims[1]
    ax1 = np.linspace(x_lims[0], x_lims[1], n, endpoint=False)
    ax2 = np.linspace(y_lims[0], y_lims[1], n, endpoint=False)
    X, Y = np.meshgrid(ax1, ax2)

    for i in range(n):
        for j in range(n):
            func[j, i] = solution.eval([ax1[i], ax2[j]], derivatives, func=func_num)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7))
    surf = ax.plot_surface(
        X, Y, func, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )

    # ax.set_xticks(X)
    # ax.set_xticks(Y)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.savefig("plot" + str(func_num) + ".pdf")
    plt.show()

