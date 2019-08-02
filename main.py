#! coding:utf-8

import pandas as pd
from climbing_irt.irt import IRT

data = pd.read_csv('data/IFSC_worldcup.csv').set_index('Name')
results = list(data.fillna(0).values)

num_climbers, num_problems = data.shape

irt = IRT(num_climbers=num_climbers, num_problems=num_problems, init_a=1.7, init_b=0.5, init_t=1.0)
irt.fit(results)

for climber, t in zip(data.index, irt.theta):
    print(climber, t)

for problem, a, b in zip(data.columns, irt.alpha, irt.beta):
    print(problem, a, b)