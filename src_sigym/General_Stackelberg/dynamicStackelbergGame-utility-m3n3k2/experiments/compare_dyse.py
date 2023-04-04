from dynamic_stackelberg.dyse import dySE_binary, dySE_fast, dySE_MILP, dySE
from dynamic_stackelberg.utils import setup_random_game_int, setup_random_game
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

K = 100
ntypes = 2
m = 3
n = 3
T = 3

def verify_milp():
    for k in range(K):
        R, C, pi = setup_random_game_int(m, n, ntypes, security=False)

        dySE_Model = dySE_binary(R, C, pi, T, False)
        dySE_util = dySE_Model.objVal

        dySE_milp_Model = dySE_MILP(R, C, pi, T, False)
        dySE_milp_util = dySE_milp_Model.objVal

        if abs(dySE_util - dySE_milp_util) > 0.00001:
            print("Failure:")
            print(R)
            print(C)
            print(pi)
            print("Fast obj: ", dySE_milp_util)
            print("True obj: ", dySE_util)

def verify_fast():
    for k in range(K):
        R, C, pi = setup_random_game_int(m, n, ntypes, security=False)

        dySE_Model = dySE_binary(R, C, pi, T, False)
        dySE_util = dySE_Model.objVal

        dySE_fast_Model = dySE_fast(R, C, pi, T, False)
        dySE_fast_util = dySE_fast_Model.objVal

        if abs(dySE_util - dySE_fast_util) > 0.00001:
            print("Failure:")
            print(R)
            print(C)
            print(pi)
            print("Fast obj: ", dySE_fast_util)
            print("True obj: ", dySE_util)

def verify_binary():
    for k in range(K):
        R, C, pi = setup_random_game_int(m, n, ntypes, security=False)

        dySE_Model = dySE_binary(R, C, pi, T, False)
        dySE_util = dySE_Model.objVal

        dySE_exp_Model = dySE(R, C, pi, T, False)
        dySE_exp_util = dySE_exp_Model.objVal

        if abs(dySE_util - dySE_exp_util) > 0.00001:
            print("Failure:")
            print(R)
            print(C)
            print(pi)
            print("Fast obj: ", dySE_exp_util)
            print("True obj: ", dySE_util)

def verify_all_dynamic():
    for k in range(K):
        R, C, pi = setup_random_game_int(m, n, ntypes, security=False)
        dySE_exp_Model = dySE(R, C, pi, T, False)
        dySE_exp_util = dySE_exp_Model.objVal

        dySE_Model = dySE_binary(R, C, pi, T, False)
        dySE_exp_var_util = dySE_Model.objVal

        dySE_fast_Model = dySE_fast(R, C, pi, T, False)
        dySE_fast_util = dySE_fast_Model.objVal

        dySE_milp_Model = dySE_MILP(R, C, pi, T, False)
        dySE_milp_util = dySE_milp_Model.objVal

        if abs(dySE_exp_util - dySE_exp_var_util) > 0.00001 or abs(dySE_exp_var_util - dySE_fast_util) > 0.00001 or abs(dySE_fast_util - dySE_milp_util) > 0.00001:
            print("Failure:")
            print(R)
            print(C)
            print(pi)
            print("objectives: ", dySE_exp_util, dySE_exp_var_util, dySE_fast_util, dySE_milp_util)
            #print("Fast obj: ", dySE_exp_util)
            #print("True obj: ", dySE_util)

def verify_pricing():
    # R = np.array([[0.5, 0.0], [1.0, 0.0]])
    # C = np.array([[[0.0, 0.0], [-0.5, 0.0]], [[0.5, 0.0], [0.0, 0.0]]])
    # pi = [0.5, 0.5]
    R = np.array([[0.4, 0.0], [0.5, 0.0], [0.6, 0.0]])
    C = np.array([[[0.0, 0.0], [-0.1, 0.0], [-0.2, 0.0]], [[0.1, 0.0], [0.0, 0.0], [-0.1, 0.0]], [[0.2, 0.0], [0.1, 0.0], [0.0, 0.0]]])
    pi = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    #base, _, _ = dySE_binary(R, C, pi, 8, False)
    base = dySE_MILP(R, C, pi, 8, False)
    base_util = base.objVal
    print(base_util)
    # for T in range(5):
    #     dySE_Model, logging_X, logging_y = dySE_binary(R, C, pi, T, False)
    #     dySE_exp_var_util = dySE_Model.objVal
    #     if abs(base_util*T - dySE_exp_var_util) > 0.00001:
    #         print("False")
    #         return
    
    
    # dySE_exp_Model = dySE(R, C, pi, T, False)
    # dySE_exp_util = dySE_exp_Model.objVal

    # dySE_Model, logging_X, logging_y = dySE_binary(R, C, pi, T, False)
    # dySE_exp_var_util = dySE_Model.objVal

    # dySE_fast_Model = dySE_fast(R, C, pi, T, False)
    # dySE_fast_util = dySE_fast_Model.objVal

    # dySE_milp_Model = dySE_MILP(R, C, pi, T, False)
    # dySE_milp_util = dySE_milp_Model.objVal
    # print("objectives: ", dySE_exp_util, dySE_exp_var_util, dySE_fast_util, dySE_milp_util)
    # print(logging_X, logging_y)
    # if abs(dySE_exp_util - dySE_exp_var_util) > 0.00001 or abs(dySE_exp_var_util - dySE_fast_util) > 0.00001 or abs(dySE_fast_util - dySE_milp_util) > 0.00001:
    #     print("Failure:")
    #     print(R)
    #     print(C)
    #     print(pi)
    #     print("objectives: ", dySE_exp_util, dySE_exp_var_util, dySE_fast_util, dySE_milp_util)

#verify_milp()
#verify_all_dynamic()
#verify_pricing()


# def verify_runtime():
#     t1, t2 = 0, 0
#     for k in range(1):
#         R, C, pi = setup_random_game_int(m, n, ntypes, security=False)
#         tt1 = time.time()
#         dySE_Model, _, _ = dySE_binary(R, C, pi, 7, False)
#         t1 += time.time() - tt1
#         dySE_exp_var_util = dySE_Model.objVal

#         #dySE_fast_Model = dySE_fast(R, C, pi, T, False)
#         #dySE_fast_util = dySE_fast_Model.objVal
#         tt2 = time.time()
#         dySE_milp_Model = dySE_MILP(R, C, pi, 7, False)
#         t2 += time.time() - tt2
#         dySE_milp_util = dySE_milp_Model.objVal
#     print(t1, t2)

def verify_server():
    for T in tqdm(range(20)):
        logging = open("logging.txt", "a")
        logging.write(str(time.asctime()) + "\n")
        logging.write(str(T) + "\n")
        t = 0.0
        R, C, pi = setup_random_game_int(m, n, ntypes, security=False)
        tt1 = time.time()
        dySE_Model = dySE_binary(R, C, pi, T, False)
        t += time.time() - tt1
        dySE_exp_var_util = dySE_Model.objVal
        logging.write(str(t)+"\n")
        #dySE_fast_Model = dySE_fast(R, C, pi, T, False)
        #dySE_fast_util = dySE_fast_Model.objVal
        # tt2 = time.time()
        # dySE_milp_Model = dySE_MILP(R, C, pi, 7, False)
        # t2 += time.time() - tt2
        # dySE_milp_util = dySE_milp_Model.objVal
    #print(t/1.0)

verify_server()
print("Success!")