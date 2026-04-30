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

import pandas as pd

name='beta'
n=9
m=5
data =np.zeros((m,n))
for i in range(m):
    for j in range(n):
        df = pd.read_csv('logs'+str(i)+'_'+str(j)+'.csv')
        print(df.head())
        data[i,j] = df.iloc[-1][name]#['residual']

print(data)
