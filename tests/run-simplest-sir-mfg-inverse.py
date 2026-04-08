from clspde.utils import plot, eval_dict
from clspde.prepare import from_file, prepare_settings
from clspde.solution import Solution
import itertools
import numpy as np
import copy
import yaml
import pickle as pkl

settings_filename = "settings/simplest_mfg.yaml"
settings, sol_mes, iteration_dict = from_file(settings_filename)
with open('colloc_solution_coefs.pkl', 'rb') as in_file:
    coefs = pkl.load(in_file)

sol_mes.cells_coefs = coefs['coefs']



settings_filename = "settings/simplest_mfg_inverse.yaml"
with open(settings_filename, mode="r") as file:
    settings = yaml.safe_load(file)



settings['CUSTOMS']['I_info'] = lambda x : sol_mes.eval(point=x, der=[0], func=1, cells_closed_right=True)
settings['DATA_POINTS'] = np.array(np.linspace(-1,1,6*30).reshape(-1,1))#utils.f_collocation_points(settings['MODEL']['power']+1)


from copy import deepcopy as cp
temp_settings = cp(settings)

print(temp_settings)
settings, iteration_dict = prepare_settings(settings)
sol = Solution(**eval_dict(settings['MODEL'], {'np':np}))
sol.cells_coefs *= 0.0


print('BETA', (sol_mes.eval([0.0001],[1],func=1)/0.3 + 2)/0.7, sol_mes.eval([0.0001],[1],func=1))

n = 20
ts = np.linspace(settings['MODEL']["area_lims"][0, 0], settings['MODEL']["area_lims"][0, 1] - 1e-9, n)
def eval_error():
    er = [0]*5
    for func in range(4):
        for t in ts:
            inc = sol.eval([t],[0],func) - sol_mes.eval([t],[0],func)
            er[func] += float(inc)**2
        er[4] = abs(sol.eval([0.2],[0],func=4)-20)
    return er


def pack_coefs(sol):
    res = np.zeros(np.prod(sol.cells_shape))
    inds = [list(range(size)) for size in sol.dim_sizes]
    all_cells = list(itertools.product(*inds))
    cell_shape = tuple([sol.power] * sol.n_dims)
    cell_size = np.prod(cell_shape)
    size = int(cell_size * sol.n_funcs)
    for cell in all_cells:
        cell_index = sol.cell_index(cell)
        cell_res = np.zeros(size)
        for i in range(sol.n_funcs):
            cell_res[i * cell_size :
               (i + 1) * cell_size] = sol.cells_coefs[(i, *cell)].ravel()
        
        res[size * cell_index : size * (cell_index + 1)] = cell_res
    return res

def eval_residuals(raw_res, name, i):
    def delete_part(settings, name):
        settings[name]['left'] = []
        settings[name]['right'] = []
        return settings
    def choose_part(settings, name, i):
        settings_left = []
        settings_right = []
        for ii in i:
            settings_left.append(settings[name]['left'][ii])
            settings_right.append(settings[name]['right'][ii])
        settings[name]['left'] = settings_left
        settings[name]['right'] = settings_right
        return settings
    inner_temp_settings = cp(temp_settings)
    names = ['COLLOC_OPS', 'BORDER_OPS']
    for n in names:
        if n != name:
            inner_temp_settings = delete_part(inner_temp_settings, n)
    inner_temp_settings = choose_part(inner_temp_settings, name, i)
    inner_temp_settings, inner_iteration_dict = prepare_settings(inner_temp_settings)
    A, b = sol.global_solve(
        alpha=0, #1e-7,
        calculate=False,
        **inner_iteration_dict,
    )
    #print('AAAAAAAAAAAA', A)
    return np.sqrt(np.sum((A @ raw_res - b)**2))/len(b)
    
print(iteration_dict)



k = 100

true_resudual = np.empty(k)*np.nan
all_errors = np.empty((k,5+1+4))*np.nan
for j in range(k):
    prev_coefs = copy.deepcopy(sol.cells_coefs)
    #prev_eval = sol_eval(sol)
    A, b = sol.global_solve(
        solver="np",
        #svd_threshold=1e-8,
        alpha=0, #1e-7,
        **iteration_dict,
    )
    speed = 0.9
    raw_res = pack_coefs(sol)
    sol.cells_coefs = (1-speed)*prev_coefs + speed*sol.cells_coefs
    true_resudual[j] = np.sqrt(np.sum((A @ raw_res - b)**2))/len(b)
    errors = eval_error()
    all_errors[j,:len(errors)] = errors
    all_errors[j,len(errors)] = true_resudual[j]
    
    for i in range(4):
        all_errors[j,len(errors)+1+i] = eval_residuals(raw_res, 'COLLOC_OPS', [i])
    #for i in range(2):
    #    all_errors[j,len(errors)+1+4+i] = eval_residuals(raw_res, 'BORDER_OPS', [i*2,i*2+1])
    #print(all_errors)
    coef_change = np.max(np.abs(prev_coefs - sol.cells_coefs))
    print(j,' | ', coef_change , ' | ', true_resudual[j],' | ', errors)
    if coef_change<1e-7:
        break
#plot(sol)

import pandas as pd
col_names = ['err_S', 'err_I', 'err_uS','err_uI', 'beta', 'residual', 'residual_S', 'residual_I', 'residual_uS', 'residual_uI',] #'residual_initial', 'residual_terminal']
logs = pd.DataFrame(all_errors, columns=col_names)
logs = logs.dropna()
logs['index']=logs.index
logs.to_csv('logs.csv', sep=',')
#params_to_save = copy.deepcopy(params)
#params_to_save.pop("basis", None)
#params_to_save["coefs"] = sol.cells_coefs

#dump_pars(pars_to_save)

n = 20
ts = np.linspace(settings['MODEL']["area_lims"][0, 0], settings['MODEL']["area_lims"][0, 1] - 1e-9, n)
points = [[t] for t in ts]
vals = [[sol.eval(np.array([t]), [0], 1)] for t in ts]

#import matplotlib.pyplot as plt
#plt.plot(points, vals)
#plt.show()

out_dict = {'points':points, 'data':vals}
with open('colloc_solution_I.pkl', 'wb') as out_file:
    pkl.dump(out_dict, out_file)
