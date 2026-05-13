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


settings_filename = "settings/simplest_mfg.yaml"
settings, sol_mes, iteration_dict = from_file(settings_filename)
with open('colloc_solution_coefs.pkl', 'rb') as in_file:
    coefs = pkl.load(in_file)

sol_mes.cells_coefs = coefs['coefs']

n=300
points = np.linspace(-1,1,n).reshape(-1)
data = np.zeros((4,n))
for i in range(4):
    for j, point in enumerate(points):
        data[i][j] = sol_mes.eval(point, [0], i)

col_names = ['S','I','pS','pI'] #'residual_initial', 'residual_terminal']
logs = pd.DataFrame(data, columns=col_names)
logs['index']=logs.index
logs.to_csv('si.csv', sep=',', float_format='%.3e')
