from dynamic_stackelberg.dyse import dySE_binary
from dynamic_stackelberg.SSE import PerfectLearning
from dynamic_stackelberg.bse import BSE_binary
from dynamic_stackelberg.utils import setup_random_game_int, setup_random_game, setup_random_game_zerosum
from dynamic_stackelberg.Stackelberg_menu import randomized_Stackelberg_menu, deterministic_Stackelberg_menu
import numpy as np
import time
from tqdm import tqdm 
import pandas as pd

K = 50
ntypes = 1
m = 2
n = 2
#T = 2


def verify_normal_game():


    columns=["BSS", "DySS", "DSM", "RSM", "R1FI", "R", "C",
            "DySS > BSS", "DySS > DSM", "DySS > RSM", "DySS > R1FI", "R1FI > DSM"]
    # columns=["BSS", "DySS", "DSM", "RSM", "R1FI", "R", "C",
    #         "DySS > BSS", "DySS > DSM", "DySS > RSM", "DySS > R1FI", "R1FI > DSM",
    #         "DySS EQ", "Perfect EQ"]
    results = pd.DataFrame([], columns=columns)

    for T in range(1, 11):
        BSS, dySE, DSM, RSM, R1FI = 0.0, 0.0, 0.0, 0.0, 0.0
        for k in range(K):
            R, C, pi = setup_random_game_int(m, n, ntypes, security=False)
            
            BSS += T*BSE_binary(R, C, pi, False).objVal
            dySE_Model = dySE_binary(R, C, pi, T, False)
            dySE += dySE_Model.objVal

            dySE_eq = []
            # for v in dySE_Model.getVars():
            #     dySE_eq += [(v.varName, v.x)]

            R = [R for _ in range(ntypes)]
            DSM += T*deterministic_Stackelberg_menu(m, n, ntypes, pi, R, C)
            RSM += T*randomized_Stackelberg_menu(m, n, ntypes, pi, R, C)
            R1FI_T, R1FI_eqs = PerfectLearning(m, n, ntypes, pi, R, C)
            R1FI += R1FI_T*T

            # BSS = T*BSE_binary(R, C, pi, False).objVal
            # dySE_Model = dySE_binary(R, C, pi, T, False)
            # dySE = dySE_Model.objVal

            # dySE_eq = []
            # # for v in dySE_Model.getVars():
            # #     dySE_eq += [(v.varName, v.x)]

            # R = [R for _ in range(ntypes)]
            # DSM = T*deterministic_Stackelberg_menu(m, n, ntypes, pi, R, C)
            # RSM = T*randomized_Stackelberg_menu(m, n, ntypes, pi, R, C)
            # R1FI_T, R1FI_eqs = PerfectLearning(m, n, ntypes, pi, R, C)
            # R1FI = R1FI_T*T
        BSS, dySE, DSM, RSM, R1FI = BSS/float(K), dySE/float(K), DSM/float(K), RSM/float(K), R1FI/float(K)
        new_row = pd.DataFrame([[BSS, dySE, DSM, RSM, R1FI, R, C,
                                (dySE > BSS), (dySE > DSM), (dySE > RSM), (dySE > R1FI), (R1FI > DSM)]], columns=columns)

        # new_row = pd.DataFrame([[BSS, dySE, DSM, RSM, R1FI, R, C,
        #                         (dySE > BSS), (dySE > DSM), (dySE > RSM), (dySE > R1FI), (R1FI > DSM),
        #                         dySE_eq, R1FI_eqs]], columns=columns)

        results = pd.concat([results, new_row])

    results.to_csv("results/comparison_suite_T5.csv")

def verify_zerosum_game():
    T = 2
    columns=["BSS", "DySS", "DSM", "RSM", "R1FI", "R", "C",
            "DySS > BSS", "DySS > DSM", "DySS > RSM", "DySS > R1FI", "R1FI > DSM",
            "DySS EQ", "Perfect EQ"]
    results = pd.DataFrame([], columns=columns)
    for k in tqdm(range(K)):
        R, C, pi = setup_random_game_zerosum(m, n, ntypes, security=False)
        print(R, C, pi)
        BSS = 0.0
        dySE = 0.0
        dySE_eq = [0.0]
        print(BSE_binary(R, C, pi, False).status)
        #BSS = T*BSE_binary(R, C, pi, False).objVal
        # dySE_Model = dySE_binary(R, C, pi, T, False)
        # dySE = dySE_Model.objVal

        # dySE_eq = []
        # for v in dySE_Model.getVars():
        #     dySE_eq += [(v.varName, v.x)]

        R = [R for _ in range(ntypes)]
        DSM = T*deterministic_Stackelberg_menu(m, n, ntypes, pi, R, C)
        RSM = T*randomized_Stackelberg_menu(m, n, ntypes, pi, R, C)
        R1FI_T, R1FI_eqs = PerfectLearning(m, n, ntypes, pi, R, C)
        R1FI = R1FI_T*T

        new_row = pd.DataFrame([[BSS, dySE, DSM, RSM, R1FI, R, C,
                                (dySE > BSS), (dySE > DSM), (dySE > RSM), (dySE > R1FI), (R1FI > DSM),
                                dySE_eq, R1FI_eqs]], columns=columns)

        results = pd.concat([results, new_row])
    results.to_csv("results/comparison_suite_T2_zerosum.csv")

def main():
    verify_normal_game
    #verify_zerosum_game()

if __name__ == "__main__":
    main()