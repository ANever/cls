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


df = pd.read_csv('logs'+str(0)+'_'+str(0)+'.csv')
names = df.columns

name='beta'
n=9
m=5
for name in names:        
    data =np.zeros((m,n))
    for i in range(m):
        string = ''
        for j in range(n):
            df = pd.read_csv('logs'+str(i)+'_'+str(j)+'.csv')
            #print(df.head())
            datum = df.iloc[-1][name]#/20
            data[i,j] = datum #df.iloc[-1][name]#['residual']
            string += f"{datum:.3f} & "
        print(string[:-2] + '\\\\')
    print(data)

    df = pd.DataFrame(data.transpose())
    df['x'] = 50*np.array(list([2**i for i in range(n)]))
    #df['x'] = np.array([0, 0.01, 0.05, 0.1, 0.2])
    df.to_csv('err_' + name + '.csv',index=False)
print(names)