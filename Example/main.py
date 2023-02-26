import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from General_Stackelberg.dynamic_stackelberg import dyse, SSE
from General_Stackelberg.dynamic_stackelberg import utils
import utils


def generate_randome_instance(trials, m, n):
    for trial in range(trials):
        defender_payoff = "random_instance/R_{}.txt".format(trial)
        attacker_payoff = "random_instance/C_{}.txt".format(trial)
        R =  np.random.rand(m, n)
        C = np.random.rand(m, n)
        np.savetxt(
            defender_payoff,
            R,
            header='columns: follower actions, rows: leader actions',
            fmt='%.6f'
        )
        np.savetxt(
            attacker_payoff,
            C,
            header='columns: follower actions, rows: leader actions',
            fmt='%.6f'
        )


def main():
    defender_mode_dict = {
        '1': "random",
        '2': 'MWU'
    }
    attacker_mode_dict = {
        '1': "random",
        '2': 'BestResponse'
    }
    defender_mode = input("Please select defender behavior mode: \n1 for random; 2 for MWU: \n")
    print("you selected " + defender_mode_dict[defender_mode] + " defender.")
    attacker_mode = input("Please select attacker behavior mode: \n1 for random; 2 for BestResponse: \n")
    print("you selected " + attacker_mode_dict[attacker_mode] + " attacker.")
    str_T = input("Please enter the number of interaction rounds: ")
    print("The total number of interaction rounds is " + str_T + ".")
    T = int(str_T)
    str_trial = input("Please enter the number of trials: ")
    print("The total number of interaction rounds is " + str_trial + ".")
    trial = int(str_trial)
    m, n = 3, 3
    eps = 0.01

    if defender_mode == '2':
        rgt = 0.0
        for tr in range(trial):
            R = np.loadtxt("random_instance/R_{}.txt".format(tr))
            C = np.loadtxt("random_instance/C_{}.txt".format(tr))
            u_sse, _ = SSE.StackelbergEquilibrium(2, 2, R, C)
            x = [1.0/m for _ in range(m)]
            u_t = 0.0
            for t in range(T):
                j_t = None
                if attacker_mode == '1':
                    j_t = np.random.choice(np.arange(n))
                else:
                    j_t = utils.best_response(x, C)
                i_t = np.random.choice(np.arange(m), p=x)
                u_t += R[i_t][j_t]
                print("At round " + str(t) + ", the leader plays action " + str(i_t) + " according to mixed strategy " + str(x) + ", the follower responses by " + str(j_t) + ".")
                print("The leader utility at current round is {}, and the cumulative leader utility until current round is {}.".format(R[i_t][j_t], u_t))
                print("==============================="*5)
                reward_t = R[:, j_t]
                x = utils.mwu_update(x=x, reward=reward_t, eps=eps)
            rgt += (u_sse*T - u_t)/T
        print("The averaged regret you get over {} trials is {}".format(trial, rgt/trial))
    
    else:
        rgt = 0.0
        for tr in range(trial):
            R = np.loadtxt("random_instance/R_{}.txt".format(tr))
            C = np.loadtxt("random_instance/C_{}.txt".format(tr))
            u_sse, _ = SSE.StackelbergEquilibrium(2, 2, R, C)
            x = [np.random.rand() for i in range(m)]
            temp = sum(x)
            x = [i/temp for i in x]
            u_t = 0.0
            for t in range(T):
                j_t = None
                if attacker_mode == '1':
                    j_t = np.random.choice(np.arange(n))
                else:
                    j_t = utils.best_response(x, C)
                i_t = np.random.choice(np.arange(m), p=x)
                u_t += R[i_t][j_t]
                print("At round " + str(t) + ", the leader plays action " + str(i_t) + " according to mixed strategy " + str(x) + ", the follower responses by " + str(j_t) + ".")
                print("The leader utility at current round is {}, and the cumulative leader utility until current round is {}.".format(R[i_t][j_t], u_t))
                print("==============================="*5)
                x = [np.random.rand() for i in range(m)]
                temp = sum(x)
                x = [i/temp for i in x]
            rgt += (u_sse*T - u_t)/T
        print("The averaged regret you get over {} trials is {}".format(trial, rgt/trial))


if __name__ == '__main__':
    main()
    #generate_randome_instance(100, 3, 3)