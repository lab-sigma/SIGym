import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from General_Stackelberg.dynamic_stackelberg import dyse, SSE
from General_Stackelberg.dynamic_stackelberg import utils
import utils
from sigym import Follower, Platform

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
    T = 100
    m, n = 5, 5
    eps = 0.01
    trial = 1

    for attacker_mode in ['1']:
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
                R = np.loadtxt("random_instance/R_{}.txt".format(tr))
                C = np.loadtxt("random_instance/C_{}.txt".format(0))
                Rr = [[0.7917250394821167, -0.020218396559357643, -0.832619845867157, -0.7781567573547363, -0.8700121641159058], [-0.08712930232286453, 0.5288949012756348, -0.832619845867157, -0.7781567573547363, -0.8700121641159058], [-0.08712930232286453, -0.020218396559357643, 0.5680445432662964, -0.7781567573547363, -0.8700121641159058], [-0.08712930232286453, -0.020218396559357643, -0.832619845867157, 0.9255966544151306, -0.8700121641159058], [-0.08712930232286453, -0.020218396559357643, -0.832619845867157, -0.7781567573547363, 0.07103605568408966]]
                Cc = [[-0.6458941102027893, 0.7151893377304077, 0.6027633547782898, 0.5448831915855408, 0.42365479469299316], [0.54881352186203, -0.4375872015953064, 0.6027633547782898, 0.5448831915855408, 0.42365479469299316], [0.54881352186203, 0.7151893377304077, -0.891772985458374, 0.5448831915855408, 0.42365479469299316], [0.54881352186203, 0.7151893377304077, 0.6027633547782898, -0.9636627435684204, 0.42365479469299316], [0.54881352186203, 0.7151893377304077, 0.6027633547782898, 0.5448831915855408, -0.3834415078163147]]
                R = np.asarray(Rr)
                C = np.asarray(Cc)
                agent = Follower(utility_matrix=C, behavior_mode=attacker_mode_dict[attacker_mode])  # Instantiate a follower class using the utility matrix and the behavior model
                env = Platform(R, C)
                u_sse = env.compute_SSE()

                x = [np.random.rand() for i in range(m)]
                temp = sum(x)
                x = [i/temp for i in x]
                rrr = []
                for t in range(T):
                    i_t, j_t = env.step(x, agent, R) # the platform takes a step based on the leader strategy and the follower model
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
                    rrr.append((u_sse*(t+1) - env.cum_u) / (t+1))
                    #print ((u_sse*(t+1) - env.cum_u) / (t+1))
                rgt += (u_sse*T - env.cum_u)/T
                print(rrr)
            print("The averaged regret you get over {} trials is {}".format(trial, rgt/trial))

if __name__ == '__main__':
    main()
    #utils.generate_randome_instance(100, 3, 3)
