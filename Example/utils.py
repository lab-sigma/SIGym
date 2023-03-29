import numpy as np
import pandas as pd
import time
from tqdm import tqdm

def compute_follower_utility(x, C, j_t):
    # input: x is a vector of leader strategy, C is the follower utility matrix, j_t is the follower action
    # output: the follower utility
    m, n = len(C), len(C[0])
    u = 0.0
    for i in range(m):
        u += x[i]*C[i][j_t]
    return u


def best_response(x, C):
    # input: x is a vector of leader strategy, C is the follower utility matrix
    # output: the follower action
    m, n = len(C), len(C[0])
    action = -1
    max_u = -np.infty
    for j in range(n):
        temp = 0.0
        for i in range(m):
            temp += x[i]*C[i][j]
        if temp > max_u:
            max_u = temp
            action = j
    return action


def quantal_response(x, C, alpha=0.8):
    # input: x is a vector of leader strategy, C is the follower utility matrix, alpha is the quantal response parameter
    # output: the follower action sampled from the quantal response distribution
    denominator = 0.0
    for j in range(len(C[0])):
        denominator += np.exp(alpha*compute_follower_utility(x, C, j))
    follower_strategy = np.zeros(len(C[0]))
    for j in range(len(C[0])):
        follower_strategy[j] = np.exp(alpha*compute_follower_utility(x, C, j))/denominator
    return np.random.choice(np.arange(len(C[0])), p=follower_strategy)


def mwu_update(x, reward, eps):
    for i in range(len(x)):
        x[i] = x[i] * (1 + eps* reward[i])
    temp = sum(x)
    for i in range(len(x)):
        x[i] /= temp
    return x


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