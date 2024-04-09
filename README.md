# sigym Package

Strategic Interation Gym is a Python library for developing and exploring different algorithms for strategically interacting in game theoretic settings. It aims to provide implementations of a variety of common environments and response mechanisms as well as an API for setting up custom environments and algorithms.

# Installation

View at: https://test.pypi.org/project/sigym/0.0.1/
    
install_requires=['numpy', 'gurobipy', 'pandas']

# Example usage

```
import numpy as np
from tqdm import tqdm
from sigym import sigym_env

T = 2
m, n = 3, 3
trials = 10

for behavior_mode in ["random", "best_response", "quantal_response", "mwu", "ftl", 'delta_suboptimal']:
    print("--------------------"*5, "Attacker mode: {}".format(behavior_mode), "--------------------"*5)
    rgt = 0.0
    for tr in tqdm(range(trials)):
        env = sigym_env.Platform(m, n, behavior_mode)
        agent = env.follower
        u_sse = env.compute_SSE()
        cur_utility = 0.0
        for t in range(T):
            x = [np.random.rand() for i in range(m)]
            temp = sum(x)
            x = [i/temp for i in x]
            i_t, j_t = env.step(x, agent)
            cur_utility += env.compute_utility(i_t, j_t)
            x = [np.random.rand() for i in range(m)]
            temp = sum(x)
            x = [i/temp for i in x]

        rgt += (u_sse*T - cur_utility)/T

    print("The averaged regret you get over {} trials is {}".format(trials, rgt/trials))
```
