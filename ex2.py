import numpy as np
from tqdm import tqdm
#from sigym import sigym_env
import sys
sys.path.append("/Users/lilianzhao/Documents/repos/SIGym/src/SIGym")
import sigym_env

T = 50
m, n = 10, 10
trials = 10
learning_rate = 0.1  # Learning rate for gradient ascent

behavior_modes = ["best_response"]

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

        # User-defined strategy profile - random initialization
        x = [np.random.rand() for i in range(m)]
        temp = sum(x)
        x = [i/temp for i in x]  # Normalize to make it a valid probability distribution
        print(x)

        # Play T rounds
        for t in range(T):
            # Simulate leader action resulting follower response 
            i_t, j_t = env.step(x, agent)
            cur_utility += env.compute_utility(i_t, j_t)
            print("i_t:{}, j_t:{}".format(i_t, j_t))
            
            # Compute gradient of the utility with respect to the strategy vector x
            gradient = np.zeros(m)
            for i in range(m):
                # Small perturbation for numerical gradient estimation
                delta = 1e-5
                x_perturbed = x.copy()
                x_perturbed[i] += delta
                temp = sum(x_perturbed)
                x_perturbed = [i/temp for i in x_perturbed]  # Ensure it remains a valid probability distribution
                #print(x_perturbed)
                
                utility_with_perturbation = env.compute_utility(*env.step(x_perturbed, agent))
                gradient[i] = (utility_with_perturbation - cur_utility) / delta
            
            # Update the strategy vector using gradient ascent
            x += learning_rate * gradient
            print(x)
            x = np.maximum(x, 0)  # Ensure non-negativity
            print(x)
            temp = sum(x)
            print("temp:{}".format(temp))
            x = [i/temp for i in x]  # Re-normalize to make it a valid probability distribution
            print(x)

        print()
        print("trial: {}, utility: {}, u_sse: {}".format(tr, cur_utility, u_sse * T))
        rgt += (u_sse*T - cur_utility)/T

    # Print average regret
    print("The averaged regret you get over {} trials is {}".format(trials, rgt/trials))
