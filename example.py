import numpy as np
from tqdm import tqdm
#from sigym import sigym_env
import sys
sys.path.append("/Users/lilianzhao/Documents/repos/SIGym/src/SIGym")
import sigym_env

T = 2
m, n = 3, 3
trials = 50

T = 50
m, n = 10, 10
trials = 100

behavior_modes = ["best_response"]
behavior_modes = ["random", "best_response", "quantal_response", "mwu", "ftl", 'delta_suboptimal']

for behavior_mode in behavior_modes:
    print("--------------------"*2, "Attacker mode: {}".format(behavior_mode), "--------------------"*2)
    
    # Sum total regret across all trials 
    rgt = 0.0
    for tr in tqdm(range(trials)):

        # Initialize an instance
        env = sigym_env.Platform(m, n, behavior_mode)
        agent = env.follower
        u_sse = env.compute_SSE()
        cur_utility = 0.0

        # User-defined strategy profile - random update as an example
        x = [np.random.rand() for i in range(m)]
        temp = sum(x)
        x = [i/temp for i in x]

        # Play T rounds
        for t in range(T):
            # Calculate follower's best response somehow
            # optimize strategy vector after each round
            #x = [1] + [0] * (m - 1)
            #x = [1/m for i in x]

            x = [np.random.rand() for i in range(m)]
            temp = sum(x)
            x = [i/temp for i in x]

            # Simulate leader action resulting follower response 
            i_t, j_t = env.step(x, agent)
            cur_utility += env.compute_utility(i_t, j_t)

        #print()
        #print("trial: {}, utility: {}, u_sse: {}".format(tr, cur_utility, u_sse * T))
        rgt += (u_sse*T - cur_utility)/T

    # Print average regret
    print("Random Update")
    print("The averaged regret you get over {} trials is {}".format(trials, rgt/trials))