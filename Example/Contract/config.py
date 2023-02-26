from tkinter import TRUE
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

cfg.n_types = 2
cfg.n_outcomes = 2
cfg.n_actions = 2
cfg.mu = np.ones(cfg.n_types) / cfg.n_types

# cfg.n_types = 2
# cfg.n_targets = 10
# cfg.type_dist_file = 'random_instance/type_dist'
# cfg.informant_covered_payoff_file = 'random_instance/informant_covered_payoff'
# cfg.informant_uncovered_payoff_file = 'random_instance/informant_uncovered_payoff'
# cfg.attacker_payoff_file = 'random_instance/attacker_payoff'
# cfg.defender_payoff_file = 'random_instance/defender_payoff'

# cfg.direct_type_dist_file = 'type_dist.txt'
# cfg.direct_informant_covered_payoff_file = 'informant_covered_payoff.txt'
# cfg.direct_informant_uncovered_payoff_file = 'informant_uncovered_payoff.txt'
# cfg.direct_attacker_payoff_file = 'attacker_payoff.txt'
# cfg.direct_defender_payoff_file = 'defender_payoff.txt'

# cfg.prob_observe = 1.0
# cfg.n_informants = 1
# cfg.n_resources = 1
# cfg.write_lp = False
# cfg.save_solutions = False
# cfg.output_dir = 'output'
# cfg.s_hat = 1