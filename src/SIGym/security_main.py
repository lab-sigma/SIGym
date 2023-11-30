import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from General_Stackelberg.dynamic_stackelberg import dyse, SSE
from General_Stackelberg.dynamic_stackelberg import utils
import utils
from securitygym import Follower, Platform

def main():

    defender_mode_dict = {
        '1': "random",
        '2': 'MWU'
    }

    attacker_mode_dict = {
        '1': "random",
        '2': 'best_response',
        '3': "quantal_response",
        '4': "mwu",
        '5': "ftl",
        '6': 'delta_suboptimal'
    }

    defender_mode = '1'
    attacker_mode = '2'
    T = 10
    m, n = 5, 2
    eps = 0.01
    trial = 10
    Res = 2
    for attacker_mode in ['1', '2', '3', '4', '5', '6']:
        print("--------------------"*5, "Attacker mode: {}".format(attacker_mode_dict[attacker_mode]), "--------------------"*5)
        if defender_mode == '2':
            rgt = 0.0
            for tr in range(trial):
                # Loading  the utility matrices of the leader and the follower 
                R = np.loadtxt("random_instance/R_{}.txt".format(tr))
                C = np.loadtxt("random_instance/C_{}.txt".format(0)) # add these utilit matrix generation to the platform

                agent = Follower(utility_matrix=C, behavior_mode=attacker_mode_dict[attacker_mode])  # Instantiate a follower class using the utility matrix and the behavior model
                env = Platform(R, C)
                u_sse = env.compute_SSE()

                x = [1.0/m for _ in range(m)]
                for t in range(T):
                    i_t, j_t = env.step(x, agent) # the platform takes a step based on the leader strategy and the follower model
                    
                    #return R[i_t][j_t] to the user
                    
                    print("At round " + str(t) + ", the leader plays action " + str(i_t) + " according to mixed strategy " + str(x) + ", the follower responses by " + str(j_t) + ".")
                    print("The leader utility at current round is {}, and the cumulative leader utility until current round is {}.".format(R[i_t][j_t], env.cum_u))
                    print("==============================="*5)
                    
                    ###################################################
                    # the user updates the leader strategy (e.g. MWU) #
                    ###################################################
                    reward_t = R[:, j_t]
                    x = utils.mwu_update(x=x, reward=reward_t, eps=eps)

                rgt += (u_sse*T - env.cum_u)/T
            print("The averaged regret you get over {} trials is {}".format(trial, rgt/trial))
        
        else:
            rgt = 0.0
            for tr in range(trial):
                # don't provide R/C, use package to generate, and they'are unknown to the user
                R = np.loadtxt("random_instance/R_s.txt")
                C = np.loadtxt("random_instance/C_s.txt")
                
                agent = Follower(utility_matrix=C, behavior_mode=attacker_mode_dict[attacker_mode])  # Instantiate a follower class using the utility matrix and the behavior model
                env = Platform(R, C)
                u_sse = env.compute_SSE()

                x = [np.random.rand() for i in range(m)]
                temp = sum(x)
                x = [i/temp for i in x]
                
                for t in range(T):
                    i_t, j_t = env.rstep(x, agent, R, Res, m) # the platform takes a step based on the leader strategy and the follower model
                    env.cum_u += R[i_t][j_t]

                    #print("At round " + str(t) + ", the leader plays action " + str(i_t) + " according to mixed strategy " + str(x) + ", the follower responses by " + str(j_t) + ".")
                    #print("The leader utility at current round is {}, and the cumulative leader utility until current round is {}.".format(R[i_t][j_t], env.cum_u))
                    #print("==============================="*5)

                    ############################################################
                    # the user updates the leader strategy (e.g. radom update) #
                    ############################################################
                    x = [np.random.rand() for i in range(m)]
                    temp = sum(x)
                    x = [i/temp for i in x]

                rgt += (u_sse*T - env.cum_u)/T
            print("The averaged regret you get over {} trials is {}".format(trial, rgt/trial))

if __name__ == '__main__':
    main()
    #utils.generate_randome_instance(100, 3, 3)
