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
        self.historical_utility = np.zeros(self.n)
        self.randomized_strategy = [1.0/self.n for _ in range(self.n)]
        self.delta = 0.1

    def response(self, leader_strategy, leader_utility):
        if self.mode == 'best_response':
            return self.best_response(leader_strategy)
        elif self.mode == 'random':
            return np.random.choice(np.arange(self.n))
        elif self.mode == "quantal_response":
            return self.quantal_response(leader_strategy)
        elif self.mode == "mwu":
            return self.mwu(leader_strategy)
        elif self.mode == "ftl":
            return self.ftl(leader_strategy)
        elif self.mode == "delta_suboptimal":
            return self.DeltaSuboptimalResponse(leader_strategy, leader_utility, self.delta)
        else:
            raise ValueError("Invalid follower behavior mode")
        
    def compute_follower_utility(self, x, j_t):
        # input: x is a vector of leader strategy, C is the follower utility matrix, j_t is the follower action
        # output: the follower utility
        m, n = len(self.C), len(self.C[0])
        u = 0.0
        for i in range(m):
            u += x[i]*self.C[i][j_t]
        return u
        
    def best_response(self, x):
        # input: x is a vector of leader strategy, C is the follower utility matrix
        # output: the follower action
        m, n = len(self.C), len(self.C[0])
        action = -1
        max_u = -np.infty
        for j in range(n):
            temp = 0.0
            for i in range(m):
                temp += x[i]*self.C[i][j]
            if temp > max_u:
                max_u = temp
                action = j
        return action
    
    def quantal_response(self, x, alpha=0.8):
        # input: x is a vector of leader strategy, C is the follower utility matrix, alpha is the quantal response parameter
        # output: the follower action sampled from the quantal response distribution
        denominator = 0.0
        for j in range(self.n):
            denominator += np.exp(alpha*self.compute_follower_utility(x, j))
        follower_strategy = np.zeros(self.n)
        for j in range(self.n):
            follower_strategy[j] = np.exp(alpha*self.compute_follower_utility(x, j))/denominator
        return np.random.choice(np.arange(self.n), p=follower_strategy)
    
    def mwu(self, x, eps=0.1):
        # input: x is a vector of leader strategy, y is a vector of follower strategy, C is the follower utility matrix, eps is the learning rate
        # output: follower response at the current round, and the mixed follower strategy
        for i in range(self.n):
            self.randomized_strategy[i] = self.randomized_strategy[i] * (1 + eps* self.compute_follower_utility(x, i))
        temp = sum(self.randomized_strategy)
        for i in range(self.n):
            self.randomized_strategy[i] /= temp
        return np.random.choice(np.arange(self.n), p=self.randomized_strategy)
    
    def ftl(self, x):
        j_t = np.argmin(self.historical_utility)
        for j in range(self.n):
            self.historical_utility[j] += self.compute_follower_utility(x, j)
        return j_t
    
    def DeltaSuboptimalResponse(self, x, R, delta=0.1):
        # input: x is a vector of leader strategy, C is the follower utility matrix, R is the leader utility matrix, delta is the suboptimality parameter
        # output: the follower action sampled according to the DeltaSuboptimalResponse
        m, n = len(self.C), len(self.C[0])
        opt_u, opt_j = -np.infty, None
        utilities_all_j = [0.0 for _ in range(n)]
        delta_responses = []
        for j in range(n):
            temp = 0.0
            for i in range(m):
                temp += x[i]*self.C[i][j]
            utilities_all_j[j] = temp
        opt_j = np.argmax(utilities_all_j)
        for j in range(n):
            if utilities_all_j[j] > utilities_all_j[opt_j] - delta:
                delta_responses.append(j)
        num_delta_responses = len(delta_responses)
        delta_response_leader_utilities = [0.0 for _ in range(num_delta_responses)]
        for j in delta_responses:
            temp = 0.0
            for i in range(m):
                temp += x[i]*R[i][j]
            delta_response_leader_utilities[j] = temp
        return_j = np.argmin(delta_response_leader_utilities)
        return delta_responses[return_j]
    
    
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

    def step(self, x, follower, R):
        i_t = np.random.choice(np.arange(self.m), p=x)
        j_t = follower.response(x, R)
        return i_t, j_t