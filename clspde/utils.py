import copy
import numpy as np
import re 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import roots_legendre


def lp(line, *args, **kwargs):
    #print(line.split('='))
    splited = line.split('=')
    if len(splited)>2:
        raise ValueError('Too much equalities in line:' + line)
    left_operator = _lp(splited[0],*args, **kwargs)
    right_operator = _lp(splited[1],*args, **kwargs)
    return [left_operator, right_operator]


def _lp(line, function_list, variable_list, customs):
    splited = line.split(' ')

    ops_stack = []

    def is_der_operator(string: str):
        if re.findall('\(d\/d..?\)', string):
            return True
        else:
            return False
        
    def apply_ops(ops_stack: list, func: str):
        dif_powers = [0]*len(variable_list)
        for op in ops_stack:
            op = op.replace('(d/d', '')
            op = op.replace(')', '')
            op = op.split('^')

            var_index = variable_list.index(op[0])
            try:
                power = op[1]
            except:
                power = 1
            dif_powers[var_index] = int(power)
        previous = ''
        if func[:2]=='&&':
            f_name = 'u_loc'
            previous = ',prev=True'
        elif func[:1]=='&':
            f_name = 'u_loc'
        else:
            f_name = 'u_bas'
        func_index = function_list.index(func.replace('&',''))
        return (f_name+'('+str(dif_powers)+', '+str(func_index)+ previous +')')

    def is_func(string:str):
        if string[:2]=='&&' and (string[2:] in function_list):
            return (True, 'prev')
        if string[:1]=='&' and (string[1:] in function_list):
            return (True, 'local')
        if string in function_list:
            return (True, 'basis')
        else:
            return (False, None)

    res = ''
    for i in range(len(splited)):
        if is_der_operator(splited[i]):
            ops_stack.append(splited[i])
        elif is_func(splited[i])[0]:
            res += (apply_ops(ops_stack, splited[i],))
            ops_stack = []
        else:
            res += splited[i]
    res = compile(res, '<string>', 'eval')
    return lambda _self, u_loc, u_bas, x, x_loc: eval(res, customs | {'sol':_self, 'u_bas': u_bas, 'u_loc': u_loc, 'x_loc': x_loc, 'x':x})


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


def concat(a: np.array, b: np.array):
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
    for i in range(len(grid)):
        func[i] = solution.eval(grid[i], [0])
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

