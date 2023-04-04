import numpy as np

from dynamic_stackelberg.dyse import dySE_binary
from dynamic_stackelberg.bse import BSE_binary
import dynamic_stackelberg.utils as utils

def run_random_BSE_and_dySE(T, ntypes):
    #R, C, pi = utils.setup_random_no_learning_auction(ntypes)
    #R, C, pi = utils.setup_random_no_learning_test1(ntypes)
    #R, C, pi = utils.setup_random_no_learning_test2(ntypes)
    R, C, pi = utils.setup_random_no_learning_test3(ntypes)

    dySE_utility = dySE_binary(R, C, pi, T, False).objVal
    BSE_utility = T * BSE_binary(R, C, pi, False).objVal

    return dySE_utility, BSE_utility

if __name__ == '__main__':
    ntypes = 10
    T = 3
    K = 10
    results = np.array([run_random_BSE_and_dySE(T, ntypes) for k in range(K)])
    dySE_results = results[:, 0]
    BSE_results = results[:, 1]
    print(results)

    if np.allclose(dySE_results, BSE_results):
        print("Found all utilities to be close")
    else:
        print("Max dySE advantage: {:.5f}", np.argmax(dySE_results - BSE_results))
        print("Max BSE advantage: {:.5f}", np.argmax(BSE_results - dySE_results))
