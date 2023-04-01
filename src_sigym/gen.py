import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from src_sigym.General_Stackelberg.dynamic_stackelberg import dyse, SSE
from src_sigym.General_Stackelberg.dynamic_stackelberg import utils as uG
import src_sigym.utils as utils
from src_sigym.sigym import Follower, Platform

defender_mode_dict = {
    '1': "random",
    '2': 'MWU'
}

attacker_mode_dict = {
    '1': "random",
    '2': 'best_response'
}

defender_mode = '1'
attacker_mode = '2'
T = 10
m, n = 3, 3 #m defender row / n attacker colomn
eps = 0.01
trial = 1
imgheight = 600
imgwidth = 600
base = 50, 50
base2 = imgwidth/2+50, 50
squareL = min((imgwidth - base[0]*2) / m, (imgheight - base[1]*2) / n )
i_res = []
j_res = []
update_cnt = 0

info = ''

def main():
    global i_res, j_res, root, myCanvas, update_cnt, caption, string_var, info
    rgt = 0.0
    update_cnt = 0
    i_res = []
    j_res = []
    R = np.loadtxt("C:/Users/liuxi/Desktop/Academics/SIGym/src_sigym/random_instance/R_{}.txt".format(0))
    C = np.loadtxt("C:/Users/liuxi/Desktop/Academics/SIGym/src_sigym/random_instance/C_{}.txt".format(0))
    agent = Follower(utility_matrix=C, behavior_mode=attacker_mode_dict[attacker_mode])  # Instantiate a follower class using the utility matrix and the behavior model
    env = Platform(R, C)
    u_sse = env.compute_SSE()
    if defender_mode == '1':
        x = [np.random.rand() for i in range(m)]
        temp = sum(x)
        x = [i/temp for i in x]
    else:
        x = [1.0/m for _ in range(m)]
    for t in range(T):
        i_t, j_t = env.step(x, agent) # the platform takes a step based on the leader strategy and the follower model
        env.cum_u += R[i_t][j_t]
        i_res.append(i_t)
        j_res.append(j_t)
        print("At round " + str(t) + ", the leader plays action " + str(i_t) + " according to " + defender_mode_dict[defender_mode] + str(x) + ", the follower responses by " + str(j_t) + ".")
        #info+="At trial " + str(tr) + " round " + str(t) + ", the leader plays action " + str(i_t) + " according to " + defender_mode_dict[defender_mode] + str(x) + ", the follower responses by " + str(j_t) + ".<br>"
        print("The leader utility at current round is {}, and the cumulative leader utility until current round is {}.".format(R[i_t][j_t], env.cum_u))
        print("==============================="*5)
        ############################################################
        # the user updates the leader strategy (e.g. random update) #
        ############################################################
        if defender_mode == '1':
            x = [np.random.rand() for i in range(m)]
            temp = sum(x)
            x = [i/temp for i in x]     
        else:
            reward_t = R[:, j_t]
            x = utils.mwu_update(x=x, reward=reward_t, eps=eps)
        #print(i_res, j_res)
        rgt += (u_sse*T - env.cum_u)/T 
    print("The averaged regret you get over {} trial(s) is {}".format(trial, rgt/trial))
    return i_res, j_res, rgt

if __name__ == '__main__':
    main()