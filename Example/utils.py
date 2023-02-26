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