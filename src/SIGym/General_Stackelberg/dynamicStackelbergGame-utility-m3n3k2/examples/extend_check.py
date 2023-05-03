import numpy as np

from dynamic_stackelberg.dyse import dySE_binary, dySE_extended_utility
from dynamic_stackelberg.utils import setup_random_game

def run_random_dySE_and_extend(m, n, T, ntypes):
    R, C, pi = setup_random_game(m, n, ntypes, security=False)

    dySE_utility = dySE_binary(R, C, pi, T, False).objVal
    extend_utility = dySE_extended_utility(R, C, pi, T, False)

    return dySE_utility, extend_utility

if __name__ == '__main__':
    ntypes = 2
    m = 2
    n = 2
    T = 4
    K = 10
    results = np.array([run_random_dySE_and_extend(m, n, T, ntypes) for k in range(K)])
    dySE_results = results[:, 0]
    BSE_results = results[:, 1]
    print(results)

    if np.allclose(dySE_results, BSE_results):
        print("Found all utilities to be close")
    else:
        print("Max dySE advantage: {:.5f}", np.max(dySE_results - BSE_results))
        print("Max BSE advantage: {:.5f}", np.max(BSE_results - dySE_results))
