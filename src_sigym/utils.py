import numpy as np
import pandas as pd
import time
from tqdm import tqdm


def best_response(x, C):
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

def setup_random_no_learning_auction(ntypes):
    V = np.random.rand(ntypes)
    R = [[0, p] for p in V]
    C = [[[0, vi - p] for p in V] for vi in V]
    pi = [1.0/ntypes for _ in range(ntypes)]
    return R, C, pi

def setup_random_no_learning_test1(ntypes):
    V = np.random.rand(ntypes)
    R = [[0, p] for p in V]
    B = [np.random.random_sample() for _ in V]
    C = [[[b, vi - p] for p in V] for vi, b in zip(V, B)]
    pi = [1.0/ntypes for _ in range(ntypes)]
    return R, C, pi

def setup_random_no_learning_test2(ntypes):
    V = np.random.rand(ntypes)
    a = np.random.random_sample()
    R = [[a, p] for p in V]
    C = [[[0, vi - p] for p in V] for vi in V]
    pi = [1.0/ntypes for _ in range(ntypes)]
    return R, C, pi

def setup_random_no_learning_test3(ntypes):
    V = np.random.rand(ntypes)
    a = np.random.random_sample()
    R = [[a, p] for p in V]
    B = [np.random.random_sample() for _ in V]
    C = [[[b, vi - p] for p in V] for vi, b in zip(V, B)]
    pi = [1.0/ntypes for _ in range(ntypes)]
    return R, C, pi

def setup_random_game(m, n, ntypes, security=True):
    if security:
        R = np.random.rand(ntypes, m, n).tolist()
    else:
        R = np.random.rand(m, n).tolist()

    C = np.random.rand(ntypes, m, n).tolist()
    pi = [1.0/ntypes for _ in range(ntypes)]
    return R, C, pi

def setup_random_game_int(m, n, ntypes, lb=0, ub=8, security=True):
    if security:
        R = np.random.randint(low=lb, high=ub, size=(ntypes, m, n))
    else:
        R = np.random.randint(low=lb, high=ub, size=(m, n))
    R = R.astype(np.float32)
    R = R.tolist()

    C = np.random.randint(low=lb, high=ub, size=(ntypes, m, n))
    C = C.astype(np.float32)
    C = C.tolist()

    pi = [1.0/ntypes for _ in range(ntypes)]
    return R, C, pi
