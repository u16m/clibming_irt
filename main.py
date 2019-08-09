#! coding:utf-8

import pandas as pd
from climbing_irt.irt import IRT

data = pd.read_csv('data/IFSC_worldcup.csv').set_index('Name')
results = list(data.fillna(0).values)

# #test
# import numpy as np
# test_climber_size = 100
# test_problem_size = 500
# data = pd.DataFrame(np.random.randint(0, 2, (test_climber_size, test_problem_size)),
#                     index=[f'climber_{i}' for i in range(test_climber_size)],
#                     columns=[f'problem_{i}' for i in range(test_problem_size)])
# results = list(data.fillna(0).values)


num_climbers, num_problems = data.shape

irt = IRT(num_climbers=num_climbers, num_problems=num_problems, init_a=1.7, init_b=0.5, init_t=0)
irt.fit(results)

for climber, t in sorted(zip(data.index, irt.theta), key=lambda x: x[1]):
    print(climber, t)

for problem, a, b in zip(data.columns, irt.alpha, irt.beta):
    print(problem, a, b)