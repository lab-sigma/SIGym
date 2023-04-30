import numpy as np
import pandas as pd
import time
from tqdm import tqdm

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