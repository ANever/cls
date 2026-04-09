from copy import deepcopy as cp
import pandas as pd
from clspde.utils import plot, eval_dict
from clspde.prepare import from_file, prepare_settings
from clspde.solution import Solution
import itertools
import numpy as np
import copy
import yaml
import pickle as pkl
from random import gauss as random

def eval_error(sol, sol_mes, A,b):
    er = [0]*10
    for func in range(4):
        for t in ts:
            inc = sol.eval([t],[0],func) - sol_mes.eval([t],[0],func)
            er[func] += float(inc)**2
        er[4] = abs(sol.eval([0.2],[0],func=4)-20)
        
    true_resudual = np.sqrt(np.sum((A @ raw_res - b)**2))/len(b)
    er[5] = true_resudual
    
    for i in range(4):
        er[5+1+i] = eval_residuals(raw_res, 'COLLOC_OPS', [i])
    #for i in range(2):
    #    all_errors[j,len(errors)+1+4+i] = eval_residuals(raw_res, 'BORDER_OPS', [i*2,i*2+1])
    #print(all_errors)
    
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
    return np.sqrt(np.sum((A @ raw_res - b)**2))/len(b)

noise_lvl_set = [0., 0.01, 0.05, 0.10, 0.20]
num_data_points_set = 45*2**np.array(range(6))

for i_noise, noise_lvl in enumerate(noise_lvl_set):
    for i_data, num_data_points in enumerate(num_data_points_set):
        
        settings_filename = "settings/simplest_mfg.yaml"
        settings, sol_mes, iteration_dict = from_file(settings_filename)
        with open('colloc_solution_coefs.pkl', 'rb') as in_file:
            coefs = pkl.load(in_file)

        sol_mes.cells_coefs = coefs['coefs']

        settings_filename = "settings/simplest_mfg_inverse.yaml"
        with open(settings_filename, mode="r") as file:
            settings = yaml.safe_load(file)

        settings['CUSTOMS']['I_info'] = lambda x : (1+random()*noise_lvl)*sol_mes.eval(point=x, der=[0], func=1, cells_closed_right=True)
        settings['DATA_POINTS'] = np.array(np.linspace(-1,1,num_data_points).reshape(-1,1))#utils.f_collocation_points(settings['MODEL']['power']+1)

        temp_settings = cp(settings)
        settings, iteration_dict = prepare_settings(settings)
        sol = Solution(**eval_dict(settings['MODEL'], {'np':np}))
        sol.cells_coefs *= 0.0

        n = 20
        ts = np.linspace(settings['MODEL']["area_lims"][0, 0], settings['MODEL']["area_lims"][0, 1] - 1e-9, n)
            
        num_of_iterations = 100
        true_resudual = np.empty(num_of_iterations)*np.nan
        all_errors = np.empty((num_of_iterations,5+1+4))*np.nan
        for j in range(num_of_iterations):
            prev_coefs = copy.deepcopy(sol.cells_coefs)
            A, b = sol.global_solve(
                solver="np",
                #svd_threshold=1e-8,
                alpha=0, #1e-7,
                **iteration_dict,
            )
            speed = 0.9
            raw_res = pack_coefs(sol)
            sol.cells_coefs = (1-speed)*prev_coefs + speed*sol.cells_coefs
            errors = eval_error(sol, sol_mes, A, b)
            all_errors[j] = errors
            
            coef_change = np.max(np.abs(prev_coefs - sol.cells_coefs))
            print(j,' | ', coef_change ,' | ', errors)
            if coef_change<1e-7:
                break
        #plot(sol)

        col_names = ['err_S', 'err_I', 'err_uS','err_uI', 'beta', 'residual', 'residual_S', 'residual_I', 'residual_uS', 'residual_uI',] #'residual_initial', 'residual_terminal']
        logs = pd.DataFrame(all_errors, columns=col_names)
        logs = logs.dropna()
        logs['index']=logs.index
        logs.to_csv('logs'+str(i_noise) + '_' + str(i_data) + '.csv', sep=',')

n = 20
ts = np.linspace(settings['MODEL']["area_lims"][0, 0], settings['MODEL']["area_lims"][0, 1] - 1e-9, n)
points = [[t] for t in ts]
vals = [[sol.eval(np.array([t]), [0], 1)] for t in ts]
