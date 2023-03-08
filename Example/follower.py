import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from General_Stackelberg.dynamic_stackelberg import dyse, SSE
from General_Stackelberg.dynamic_stackelberg import utils
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