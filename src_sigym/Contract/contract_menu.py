import logging
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random

def cartesian_product_transpose(*arrays):
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = np.prod(broadcasted[0].shape), len(broadcasted)
    dtype = np.result_type(*arrays)

    out = np.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T


def randomized_contract_menu(mu, r, P, c):
    """
    m: num of outcomes
    n: number of agent actions
    k: numer of agent types
    mu: prior distribution over agent types
    P: shape of [k, n, m], probabilistic distribution over outcomes of all agent types
    c: shape of [k, n], cost over every action of all agent types
    """
    k = len(mu)
    
    m = len(r)
    n = len(c[0])

    assert len(P) == k
    assert len(P[0]) == n
    assert len(P[0][0]) == m

    model = gp.Model('bilinear')
    model.Params.NonConvex = 2

    pi = []
    for theta in range(k):
        pi_theta = []
        for a in range(n):
            pi_theta += [model.addVar(lb=0.0, ub=1.0, name='pi_{}_{}'.format(theta, a))]
        model.update()
        model.addConstr(gp.quicksum(pi_theta) == 1.0, name="pi_{}_sum".format(theta))
        pi.append(pi_theta)

    X = []
    for theta in range(k):
        x_theta = []
        for a in range(n):
            x_theta_a = []
            for i in range(m):
                x_theta_a += [model.addVar(lb=0.0, name='x_{}_{}_{}'.format(theta, a, i))]
            model.update()
            x_theta.append(x_theta_a)
        X.append(x_theta)

    for theta in range(k):
        agent_utility = 0.0
        for a in range(n):
            agent_utility_a = 0.0
            for i in range(m):
                agent_utility_a += P[theta][a][i] * X[theta][a][i]
            agent_utility_a -= c[theta][a]
            agent_utility_a *= pi[theta][a]
            agent_utility += agent_utility_a
        
        deceptions = [np.arange(n)]
        all_decep = cartesian_product_transpose(*(deceptions*n))
        for delta_index, delta in enumerate(all_decep):
            for theta_prime in range(k):
                agent_utility_prime = 0.0
                for a_prime in range(n):
                    agent_utility_prime_a_prime = 0.0
                    for i in range(m):
                        agent_utility_prime_a_prime += X[theta_prime][a_prime][i] * P[theta][delta[a_prime]][i]
                    agent_utility_prime_a_prime -= c[theta][delta[a_prime]]
                    agent_utility_prime_a_prime *= pi[theta_prime][a_prime]
                    agent_utility_prime += agent_utility_prime_a_prime
                model.addConstr(agent_utility >= agent_utility_prime, name='ic_{}_{}_{}_{}_{}'.format(theta, a, delta_index, theta_prime, a_prime))

    objective = 0.0
    for theta in range(k):
        theta_obj = 0.0
        for a in range(n):
            a_obj = 0.0
            for i in range(m):
                a_obj += (r[i] - X[theta][a][i]) * P[theta][a][i]
            a_obj *= pi[theta][a]
            theta_obj += a_obj
        theta_obj *= mu[theta]
        objective += theta_obj
    model.setObjective(objective, GRB.MAXIMIZE)

    model.optimize()

    logging_pi, logging_X = [], []
    
    for theta in range(k):
        logging_pi_theta = []
        for a in range(n):
            logging_pi_theta += [pi[theta][a].x]#[model.addVar(lb=0.0, ub=1.0, name='pi_{}_{}'.format(theta, a))]
        #model.update()
        #model.addConstr(gp.quicksum(pi_theta) == 1.0, name="pi_{}_sum".format(theta))
        logging_pi.append(logging_pi_theta)
    
    for theta in range(k):
        logging_x_theta = []
        for a in range(n):
            logging_x_theta_a = []
            for i in range(m):
                logging_x_theta_a += [X[theta][a][i].x]#[model.addVar(lb=0.0, name='x_{}_{}_{}'.format(theta, a, i))]
            #model.update()
            logging_x_theta.append(logging_x_theta_a)
        logging_X.append(logging_x_theta)

    return model.ObjVal, logging_X, logging_pi
    

def deterministic_contract_menu(mu, r, P, c):
    """
    m: num of outcomes
    n: number of agent actions
    k: numer of agent types
    mu: prior distribution over agent types
    P: shape of [k, n, m], probabilistic distribution over outcomes of all agent types
    c: shape of [k, n], cost over every action of all agent types
    """

    k = len(mu)
    
    m = len(r)
    n = len(c[0])

    assert len(P) == k
    assert len(P[0]) == n
    assert len(P[0][0]) == m

    agent_action = [np.arange(n)]
    agent_actions = cartesian_product_transpose(*(agent_action*k))
    res = 0.0
    loggingX = []
    loggingA = []
    for action_index, action in enumerate(agent_actions):
        model = gp.Model('deterministic_contract_menu_{}'.format(action_index))
        X = []
        for theta in range(k):
            x_theta = []
            for i in range(m):
                x_theta += [model.addVar(lb=0.0, name='x_{}_{}'.format(theta, i))]
            model.update()
            X.append(x_theta)
        
        for theta in range(k):
            theta_utility = 0.0
            for i in range(m):
                theta_utility += P[theta][action[theta]][i] * X[theta][i]
            theta_utility -= c[theta][action[theta]]

            for theta_prime in range(k):
                for a_prime in range(n):
                    theta_prime_a_prime_utility = 0.0
                    for i in range(m):
                        theta_prime_a_prime_utility += P[theta][a_prime][i] * X[theta_prime][i]
                    theta_prime_a_prime_utility -= c[theta][a_prime]
                    model.addConstr(theta_utility >= theta_prime_a_prime_utility, name='ic_{}_{}_{}'.format(theta, theta_prime, a_prime))
        
        objective = 0.0
        for theta in range(k):
            theta_obj = 0.0
            for i in range(m):
                theta_obj += (r[i] - X[theta][i]) * P[theta][action[theta]][i]
            theta_obj *= mu[theta]
            objective += theta_obj
        model.setObjective(objective, GRB.MAXIMIZE)
        model.optimize()
        if model.Status == 2: 
            if model.ObjVal > res:
                res = model.ObjVal
                loggingX = []
                loggingA = action
                for theta in range(k):
                    logging_x_theta = []
                    for logging_i in range(m):
                        logging_x_theta += [X[theta][logging_i].x]
                    loggingX.append(logging_x_theta)

    return res, loggingX, loggingA

if __name__ == '__main__':

    # bug = 0.0
    # info = []
    # for b in range(1000):

    #     # outcomes = m; agent_actions = n; agent_types = k
    #     m, n, k = 3, 3, 3
    #     # outcome reward sorted in ascending order
    #     r = np.array(sorted(random.sample(range(10), m)))
    #     # uniform prior distribution for agent types
    #     mu = [1.0/k] * k

    #     P = [] # probabilistic distribution over outcomes
    #     c = [] # cost vectors
    #     for i in range(k):
    #         type_i = np.random.rand(n, m)
    #         type_i = np.apply_along_axis(lambda x: x - (np.sum(x) - 1)/len(x), 1, type_i)
    #         P.append(type_i)
    #         expected_reward = np.matmul(type_i, r)
    #         c.append(expected_reward/random.sample(range(2, 6), 1))
    #     P, c = np.array(P), np.array(c)

    #     r1 = randomized_contract_menu(m, n, k, r, mu, P, c)
    #     r2 = deterministic_contract_menu(m, n, k, r, mu, P, c)
    #     if r1 < r2: 
    #         bug += 1.0
    #         info.append([r1, r2])

    # print("!"*50, bug, info)

    # m, n, k = 3, 4, 2
    # r = np.array([0, 10, 30])
    # P = np.array(
    #     [[[1.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0],
    #     [0.0, 0.5, 0.5],
    #     [0.0, 0.0, 1.0]],
    #     [[1.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0],
    #     [0.0, 0.5, 0.5],
    #     [0.0, 0.0, 1.0]]]
    # )
    # c = np.array(
    #     [
    #         [0.0, 1.0, 3.0, 10.0],
    #         [0.0, 4.0, 12.0, 40.0]
    #     ]
    # )

    # delta = 0.02, eps = 0.01

    m, n, k = 4, 4, 2
    r = np.array([0.0, 0.0, 0.0, 50.0])
    P = np.array(
        [[[1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0 - 0.25*0.02, 0.0, 0.25*0.02],
        [0.0, 0.5 - 0.02, 0.5, 0.02],
        [0.0, 0.0, 1.0 - 0.02 - 0.02*0.02, 0.02 + 0.02*0.02]],
        [[1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0 - 0.25*0.02, 0.0, 0.25*0.02],
        [0.0, 0.5 - 0.02, 0.5, 0.02],
        [0.0, 0.0, 1.0 - 0.02 - 0.02*0.02, 0.02 + 0.02*0.02]]]
    )
    c = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 50.0]
        ]
    )
    mu1 = np.array([0.0, 1.0])
    mu2 = np.array([0.01, 0.99])
    r_mu1, r_x1, r_pi1 = randomized_contract_menu(m, n, k, r, mu1, P, c)
    d_mu1, x1, a1 = deterministic_contract_menu(m, n, k, r, mu1, P, c)
    r_mu2, r_x2, r_pi2 = randomized_contract_menu(m, n, k, r, mu2, P, c)
    d_mu2, x2, a2 = deterministic_contract_menu(m, n, k, r, mu2, P, c)
    print('Randomized contract utility: ', r_mu1, r_mu2)
    print('Deterministic contract utility: ', d_mu1, d_mu2)

    # m, n, k = 4, 4, 1
    # r = np.array([0.0, 0.0, 0.0, 50.0])
    # P = np.array(
    #     [[[1.0, 0.0, 0.0, 0.0],
    #     [0.0, 1.0 - 0.25*0.02, 0.0, 0.25*0.02],
    #     [0.0, 0.5 - 0.02, 0.5, 0.02],
    #     [0.0, 0.0, 1.0 - 0.02 - 0.02*0.02, 0.02 + 0.02*0.02]]]
    # )
    # c = np.array(
    #     [
    #         [0.0, 0.0, 0.5, 50.0]
    #     ]
    # )
    # mu1 = np.array([1.0])
    # #mu2 = np.array([0.01, 0.99])
    # #r_mu1 = randomized_contract_menu(m, n, k, r, mu1, P, c)
    # d_mu1, x1, a1 = deterministic_contract_menu(m, n, k, r, mu1, P, c)
    # # r_mu2 = randomized_contract_menu(m, n, k, r, mu2, P, c)
    # # d_mu2, x2, a2 = deterministic_contract_menu(m, n, k, r, mu2, P, c)
    # # print('Randomized contract utility: ', r_mu1, r_mu2)
    # # print('Deterministic contract utility: ', d_mu1, d_mu2)
    # print(d_mu1)