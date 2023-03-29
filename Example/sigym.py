import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from General_Stackelberg.dynamic_stackelberg import dyse, SSE
from General_Stackelberg.dynamic_stackelberg import utils
from General_Stackelberg.dynamic_stackelberg.bse import BSE_binary
from General_Stackelberg.dynamic_stackelberg.Stackelberg_menu import randomized_Stackelberg_menu
import utils

class Follower:
    def __init__(self, utility_matrix, behavior_mode) -> None:
        self.C = utility_matrix
        self.mode = behavior_mode
        self.n = len(utility_matrix[0])

    def response(self, leader_strategy):
        if self.mode == 'best_response':
            return utils.best_response(leader_strategy, self.C)
        elif self.mode == 'randome':
            return np.random.choice(np.arange(self.n))
        
class Platform:
    def __init__(self, R, C) -> None:
        self.R = R
        self.C = C
        self.num_types = 1
        self.m, self.n = len(self.R), len(self.R[0])
        self.pi = [1.0/self.num_types for _ in range(self.num_types)]
        self.cum_u = 0.0

    def compute_SSE(self):
        sse_u, vars = SSE.StackelbergEquilibrium(self.m, self.n, self.R, self.C)
        return sse_u
    
    def compute_BSE(self):
        bse_model = BSE_binary(self.R, self.C, self.pi, False)
        return bse_model.objVal
    
    def compute_RME(self):
        RME_model = randomized_Stackelberg_menu(self.m, self.n, self.num_types, self.pi, self.R, self.C)
        return RME_model.objVal

    def step(self, x, follower):
        i_t = np.random.choice(np.arange(self.m), p=x)
        j_t = follower.response(x)
        return i_t, j_t