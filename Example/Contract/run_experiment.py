import gurobipy as gp
import itertools
import numpy as np
import config
import random
import contract_menu
import dynamic_contract
from tqdm import tqdm
import bayesian_contract
import pandas as pd

def generate_random_instance():
    cfg = config.cfg
    m, n, k = cfg.n_outcomes, cfg.n_actions, cfg.n_types
    r = np.array(sorted(random.sample(range(10), m)))
    mu = np.array([1.0/k] * k)
    P = [] # probabilistic distribution over outcomes
    c = [] # cost vectors
    for i in range(k):
        type_i = np.random.rand(n, m)
        type_i = np.apply_along_axis(lambda x: x - (np.sum(x) - 1)/len(x), 1, type_i)
        P.append(type_i)
        expected_reward = np.matmul(type_i, r)
        c.append(expected_reward/random.sample(range(2, 6), 1))
    P, c = np.array(P), np.array(c)
    return mu, r, P, c

def main():
    logging = open("logging.txt", 'w')
    for t in tqdm(range(1)):
        # r = np.array([0.0, 0.0, 0.0, 50.0])
        # P = np.array(
        #     [[[1.0, 0.0, 0.0, 0.0],
        #     [0.0, 1.0 - 0.25*0.02, 0.0, 0.25*0.02],
        #     [0.0, 0.5 - 0.02, 0.5, 0.02],
        #     [0.0, 0.0, 1.0 - 0.02 - 0.02*0.02, 0.02 + 0.02*0.02]],
        #     [[1.0, 0.0, 0.0, 0.0],
        #     [0.0, 1.0 - 0.25*0.02, 0.0, 0.25*0.02],
        #     [0.0, 0.5 - 0.02, 0.5, 0.02],
        #     [0.0, 0.0, 1.0 - 0.02 - 0.02*0.02, 0.02 + 0.02*0.02]]]
        # )
        # c = np.array(
        #     [
        #         [0.0, 0.0, 0.0, 0.0],
        #         [0.0, 0.0, 0.5, 50.0]
        #     ]
        # )
        # mu = np.array([0.0, 1.0])
        
        # r = np.array([0.0, 6.0])
        # P = np.array(
        #     [[[0.61711988, 0.38288012],
        #     [0.8452521,  0.1547479]],
        #     [[0.41468667, 0.58531333],
        #     [0.77501402, 0.22498598]]]
        # )
        # c = np.array(
        #     [
        #         [0.76576024, 0.3094958 ],
        #         [0.87796999, 0.33747898]
        #     ]
        # )
        r = np.array([0.0, 6.0])
        P = np.array(
            [[[0.6, 0.4],
            [0.85,  0.15]],
            [[0.4, 0.6],
            [0.77, 0.23]]]
        )
        c = np.array(
            [
                [0.76576024, 0.3094958 ],
                [0.87796999, 0.33747898]
            ]
        )
        mu = np.array([0.5, 0.5])
        #mu, r, P, c = generate_random_instance()
        deterministic_utility, dx, da = contract_menu.deterministic_contract_menu(mu, r, P, c)
        dynamic_model, dyX, dyY, dyZ = dynamic_contract.dyCo(mu, r, P, c, 2)
        dynamic_utility = dynamic_model.objVal
        bayesian_model = bayesian_contract.BSE_binary(mu, r, P, c)
        bayesian_utility = bayesian_model.objVal
        r_mu1, r_x1, r_pi1 = contract_menu.randomized_contract_menu(mu, r, P, c)
        if True:
            logging.write(str(bayesian_utility*2)+" "+ str(dynamic_utility)+" "+ str(deterministic_utility*2)+" "+ str(r_mu1*2)+"\n")
            logging.write("mu: " + str(mu) + "\n")
            logging.write("r: " + str(r) + "\n")
            logging.write("P: " + str(P) + "\n")
            logging.write("c: " + str(c) + "\n")
            logging.write(str(deterministic_utility) + "\n")
            logging.write(str(dx)+"\n")
            logging.write(str(da)+"\n")
            logging.write(str(dynamic_utility) + "\n")
            logging.write(str(dyX)+"\n")
            logging.write(str(dyY)+"\n")
            logging.write(str(dyZ)+"\n")
            logging.write(str(r_mu1) + "\n")
            logging.write(str(r_x1)+"\n")
            logging.write(str(r_pi1)+"\n")
    logging.close()


def generate_csv():
    K = 100
    T = 2
    columns=["Bayesian_contract", "Dynamic_contract", "Deterministic_menu", "Randomized_menu", "r", "P", "c", "DC > BC","DC > DM", "DC > RM", "Perfect EQ"]

    results = pd.DataFrame([], columns=columns)
    for k in tqdm(range(K)):
        mu, r, P, c = generate_random_instance()
        
        BSS = T*bayesian_contract.BSE_binary(mu, r, P, c).objVal
        dySE_Model, _, _, _ = dynamic_contract.dyCo(mu, r, P, c, T)
        dySE = dySE_Model.objVal
        dySE_eq = []
        for v in dySE_Model.getVars():
            dySE_eq += [(v.varName, v.x)]

        DSM, _, _ = contract_menu.deterministic_contract_menu(mu, r, P, c)
        RSM, _, _ = contract_menu.randomized_contract_menu(mu, r, P, c)

        DSM *= T
        RSM *= T

        new_row = pd.DataFrame([[BSS, dySE, DSM, RSM, r, P, c, (dySE > BSS), (dySE > DSM), (dySE > RSM),dySE_eq]], columns=columns)

        results = pd.concat([results, new_row])

    results.to_csv("results/comparison_suite_T2.csv")

if __name__ == '__main__':
    main()
    #generate_csv()