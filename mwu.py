import numpy as np
from tqdm import tqdm
#from sigym import sigym_env
import sys
sys.path.append("/Users/lilianzhao/Documents/repos/SIGym/src/SIGym")
import sigym_env

T = 50
m, n = 10, 10
trials = 100
learning_rate = .9  # learning rate for MWU

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

        # Initialize uniform strategy vector
        x = np.ones(m) / m

        # Play T rounds
        for t in range(T):
            # Simulate leader action resulting follower response 
            i_t, j_t = env.step(x, agent)
            round_utility = env.compute_utility(i_t, j_t)
            cur_utility += round_utility

            # Update strategy vector using MWU with exponential scaling
            x[i_t] *= np.exp(learning_rate * round_utility)

            # Update strategy vector using MWU
            #x[i_t] *= (1 + learning_rate * round_utility)
            #x = x * (1 + learning_rate * round_utility)
            #x = np.maximum(x, 0)  # Ensure non-negativity

            # Normalize for valid probability distribution
            x /= x.sum() 
            #print(x)

        #print()
        #print("trial: {}, utility: {}, u_sse: {}".format(tr, cur_utility, u_sse * T))
        rgt += (u_sse*T - cur_utility)/T

    # Print average regret
    print("MWU with exponential scaling, learning rate:{}".format(learning_rate))
    print("The averaged regret you get over {} trials is {}".format(trials, rgt/trials))
