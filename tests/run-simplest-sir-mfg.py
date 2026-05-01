from clspde.utils import plot
from clspde.prepare import from_file
import itertools
import numpy as np
import copy
import yaml
import pickle as pkl
import matplotlib.pyplot as plt

settings_filename = "settings/simplest_mfg.yaml"
settings, sol, iteration_dict = from_file(settings_filename)

k = 50


true_resudual = np.zeros(k)
for j in range(k):
    prev_coefs = copy.deepcopy(sol.cells_coefs)
    #prev_eval = sol_eval(sol)
    A, b = sol.global_solve(
        solver="np",
        #svd_threshold=1e-8,
        alpha=0,
        **iteration_dict,
    )
    speed = 0.5
    sol.cells_coefs = (1-speed)*prev_coefs + speed*sol.cells_coefs
    raw_res = np.linalg.solve(A.T @A, A.T @b)
    true_resudual[j] = np.sqrt(np.sum((A @ raw_res - b)**2))/len(b)
    coef_change = np.max(np.abs(prev_coefs - sol.cells_coefs))
    print(j,' | ', coef_change ,' | ', true_resudual)
    if coef_change < 1e-7:
        break
plot(sol)



n = 20
for n in 45*np.array([2**i for i in range(10)]):
    ts = np.linspace(settings['MODEL']["area_lims"][0, 0], settings['MODEL']["area_lims"][0, 1] - 1e-9, n)
    points = [[t] for t in ts]
    vals = [[sol.eval(np.array([t]), [0], 1)] for t in ts]

    out_dict = {'points':points, 'data':vals}
    with open('colloc_solution_I_'+str(n)+'.pkl', 'wb') as out_file:
        pkl.dump(out_dict, out_file)

out_dict = {'coefs':sol.cells_coefs}
with open('colloc_solution_coefs.pkl', 'wb') as out_file:
    pkl.dump(out_dict, out_file)
