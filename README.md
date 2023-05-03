# SIGym

*TODO: Expand this as the bounds of the project are defined; currently in stealth mode :)*

Strategic Interation Gym is a Python library for developing and exploring different algorithms for strategically interacting in game theoretic settings.
It aims to provide implementations of a variety of common environments and response mechanisms as well as an API for setting up custom environments and
algorithms.

To install the package, run `pip install -i https://test.pypi.org/simple/ SIGym-MinbiaoHan`. An example usage:
```
import numpy as np
from src.SIGym.sigym import Follower, Platform
from src.SIGym.General_Stackelberg.dynamic_stackelberg import dyse, SSE, utils


T = 10
m, n = 3, 3
trial = 5

for attacker_mode in ["random", "best_response", "quantal_response", "mwu", "ftl", 'delta_suboptimal']:
    print("--------------------"*5, "Attacker mode: {}".format(attacker_mode), "--------------------"*5)
    rgt, cur_utility = 0.0, 0.0
    for tr in range(trial):

        # generate a random game
        R, C, _ = utils.setup_random_game_int(m, n , ntypes=1)

        R, C = R[0], C[0]
        agent = Follower(utility_matrix=C, behavior_mode=attacker_mode)  # Instantiate a follower class using the utility matrix and the behavior model
        env = Platform(R, C) # Instantiate a platform class using the reward matrix and the utility matrix
        u_sse = env.compute_SSE()

        x = [np.random.rand() for i in range(m)]
        temp = sum(x)
        x = [i/temp for i in x]
        for t in range(T):
            i_t, j_t = env.step(x, agent, R) # the platform takes a step based on the leader strategy and the follower model
            cur_utility += R[i_t][j_t]
            x = [np.random.rand() for i in range(m)]
            temp = sum(x)
            x = [i/temp for i in x]

        rgt += (u_sse*T - cur_utility)/T
    print("The averaged regret you get over {} trials is {}".format(trial, rgt/trial))
```

Ported utilities/algorithmic implementations from here: https://github.com/lab-sigma/dynamic-stackelberg

You will need Gurobi (https://www.gurobi.com/) to run the code. Licenses can be obtained for free for academic purposes.

# Follower
We provide a number of attacker models for users to test their designed strategy in a Stackelberg game.

Random: arbitrarily returns a decision in each round.

Best: returns the optimal decision under a perfect Stackelberg equilibrium.

Quantal: under a quantal response equilibrium

MWU: returns the optimal decision using multiplicative weights update.

# Platform
The Platform class provides interfaces that help users evaluate the performance of their strategy. We provide three approaches of metrics:
SSE 
BSE
RME
Platform.step(): return the response for both sides
