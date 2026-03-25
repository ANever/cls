from .solution import Solution
from .utils import eval_dict
import yaml
import re 
import numpy as np

def prepare_settings(settings):
    settings['MODEL']['n_dims'] = len(settings['IN_VAR_NAMES'])
    settings['MODEL']['n_funcs'] = len(settings['OUT_VAR_NAMES'])

    customs = settings['CUSTOMS']
    for key in customs.keys():
        if isinstance(customs[key], str):
            for i, var in enumerate(settings['OUT_VAR_NAMES']):
                 customs[key] = customs[key].replace(' '+var+' ', 'u_(func='+str(i)+')')
            compile(customs[key], '<string>', 'eval')
            customs[key] = eval(customs[key], locals() | customs | {'np':np})
    
    def lp(line, function_list=settings['OUT_VAR_NAMES'], variable_list = settings['IN_VAR_NAMES'], customs=customs):
            res = _lp(line, function_list=function_list, variable_list=variable_list, customs=customs)
            print('res', res)
            return res
            
    for cond in ['COLLOC_OPS', 'BORDER_OPS']:
        for side in ['left', 'right']:
            for i, line in enumerate(settings[cond][side]):
                settings[cond][side][i] = lp(line)

    colloc_ops = list(settings['COLLOC_OPS'].values())
    border_ops = list(settings['BORDER_OPS'].values())

    connect_points = np.array(eval(settings['CONNECT_POINTS']))
    border_points = connect_points

    power = settings['MODEL']['power']
    colloc_points = np.reshape(np.linspace(-1,1,power+2), (power+2,1))
    points = [colloc_points, connect_points, border_points]

    iteration_dict = {
        "points": points,
        "colloc_ops": colloc_ops,
        "border_ops": border_ops,
        #"connect_ops": connect_ops,
    }

    return settings, iteration_dict

def from_file(settings_filename):
    with open(settings_filename, mode="r") as file:
        settings = yaml.safe_load(file)
    settings, iteration_dict = prepare_settings(settings)
    sol = Solution(**eval_dict(settings['MODEL'], {'np':np}))
    sol.cells_coefs *= 0.0
    return settings, sol, iteration_dict


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
    
    print(res)
    res = compile(res, '<string>', 'eval')
    return lambda _self, u_loc, u_bas, x, x_loc: eval(res, customs | {'sol':_self, 
                                                 'u_bas': u_bas, 'u_loc': u_loc, 
                                                 'x_loc': x_loc, 'x':x, 'np':np})


def __lp(line, **kwargs):
    #print(line.split('='))
    splited = line.split('=')
    if len(splited)>2:
        raise ValueError('Too much equalities in line:' + line)
    left_operator = _lp(splited[0], **kwargs)
    right_operator = _lp(splited[1], **kwargs)
    return [left_operator, right_operator]
