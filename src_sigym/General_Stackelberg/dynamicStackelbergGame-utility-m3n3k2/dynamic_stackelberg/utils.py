import numpy as np

def setup_random_no_learning_auction(ntypes):
    V = np.random.rand(ntypes)
    R = [[0, p] for p in V]
    C = [[[0, vi - p] for p in V] for vi in V]
    pi = [1.0/ntypes for _ in range(ntypes)]
    return R, C, pi

def setup_random_no_learning_test1(ntypes):
    V = np.random.rand(ntypes)
    R = [[0, p] for p in V]
    B = [np.random.random_sample() for _ in V]
    C = [[[b, vi - p] for p in V] for vi, b in zip(V, B)]
    pi = [1.0/ntypes for _ in range(ntypes)]
    return R, C, pi

def setup_random_no_learning_test2(ntypes):
    V = np.random.rand(ntypes)
    a = np.random.random_sample()
    R = [[a, p] for p in V]
    C = [[[0, vi - p] for p in V] for vi in V]
    pi = [1.0/ntypes for _ in range(ntypes)]
    return R, C, pi

def setup_random_no_learning_test3(ntypes):
    V = np.random.rand(ntypes)
    a = np.random.random_sample()
    R = [[a, p] for p in V]
    B = [np.random.random_sample() for _ in V]
    C = [[[b, vi - p] for p in V] for vi, b in zip(V, B)]
    pi = [1.0/ntypes for _ in range(ntypes)]
    return R, C, pi

def setup_random_game(m, n, ntypes, security=True):
    if security:
        R = np.random.rand(ntypes, m, n).tolist()
    else:
        R = np.random.rand(m, n).tolist()

    C = np.random.rand(ntypes, m, n).tolist()
    pi = [1.0/ntypes for _ in range(ntypes)]
    return R, C, pi

def setup_random_game_int(m, n, ntypes, lb=0, ub=8, security=True):
    if security:
        R = np.random.uniform(low=0.0, high=1.0, size=(ntypes, m, n))
        #R = np.random.randint(low=lb, high=ub, size=(ntypes, m, n))
    else:
        R = np.random.uniform(low=0.0, high=1.0, size=(m, n))
        #R = np.random.randint(low=lb, high=ub, size=(m, n))
    R = R.astype(np.float32)
    R = R.tolist()

    C = np.random.uniform(low=0.0, high=1.0, size=(ntypes, m, n))
    #C = np.random.randint(low=lb, high=ub, size=(ntypes, m, n))
    C = C.astype(np.float32)
    C = C.tolist()

    pi = [1.0/ntypes for _ in range(ntypes)]
    return R, C, pi

def setup_random_game_zerosum(m, n, ntypes, lb=-8, ub=8, security=True):
    # if security:
    #     R = np.random.randint(low=lb, high=ub, size=(ntypes, m, n))
    # else:
    #     R = np.random.randint(low=lb, high=ub, size=(m, n))
    # R = R.astype(np.float32)
    # R = R.tolist()

    C = np.random.randint(low=lb, high=ub, size=(ntypes, m, n))
    R = -C
    R = R[0]
    C = C.astype(np.float32)
    C = C.tolist()
    R = R.astype(np.float32)
    R = R.tolist()
    

    pi = [1.0/ntypes for _ in range(ntypes)]
    return R, C, pi