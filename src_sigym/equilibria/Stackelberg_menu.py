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


def randomized_Stackelberg_menu(m, n, k, mu, A, B):
    """
    m: num of leader actions
    n: number of follower actions
    k: numer of agent types
    mu: prior distribution over agent types
    A: shape of [k, m, n], leader payoff matrix
    B: shape of [k, m, n], follower payoff matrix
    """

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()

    model = gp.Model('bilinear', env=env)
    model.Params.NonConvex = 2

    pi = []
    for theta in range(k):
        pi_theta = []
        for j in range(n):
            pi_theta += [model.addVar(lb=0.0, ub=1.0, name='pi_{}_{}'.format(theta, j))]
        model.update()
        model.addConstr(gp.quicksum(pi_theta) == 1.0, name="pi_{}_sum".format(theta))
        pi.append(pi_theta)

    X = []
    for theta in range(k):
        x_theta = []
        for j in range(n):
            x_theta_j = []
            for i in range(m):
                x_theta_j += [model.addVar(lb=0.0, ub=1.0, name='x_{}_{}_{}'.format(theta, j, i))]
            model.update()
            model.addConstr(gp.quicksum(x_theta_j) == 1.0, name="x_{}_{}_sum".format(theta, j))
            x_theta.append(x_theta_j)
        X.append(x_theta)

    for theta in range(k):
        follower_utility = 0.0
        for j in range(n):
            follower_utility_j = 0.0
            for i in range(m):
                follower_utility_j += B[theta][i][j] * X[theta][j][i]
            follower_utility_j *= pi[theta][j]
            follower_utility += follower_utility_j
        
        deceptions = [np.arange(n)]
        all_decep = cartesian_product_transpose(*(deceptions*n))
        for delta_index, delta in enumerate(all_decep):
            for theta_prime in range(k):
                follower_utility_prime = 0.0
                for j_prime in range(n):
                    follower_utility_prime_j_prime = 0.0
                    for i in range(m):
                        follower_utility_prime_j_prime += X[theta_prime][j_prime][i] * B[theta][i][delta[j_prime]]
                    follower_utility_prime_j_prime *= pi[theta_prime][j_prime]
                    follower_utility_prime += follower_utility_prime_j_prime
                model.addConstr(follower_utility >= follower_utility_prime, name='ic_{}_{}_{}_{}_{}'.format(theta, j, delta_index, theta_prime, j_prime))

    objective = 0.0
    for theta in range(k):
        theta_obj = 0.0
        for j in range(n):
            j_obj = 0.0
            for i in range(m):
                j_obj += X[theta][j][i] * A[theta][i][j]
            j_obj *= pi[theta][j]
            theta_obj += j_obj
        theta_obj *= mu[theta]
        objective += theta_obj
    model.setObjective(objective, GRB.MAXIMIZE)

    model.optimize()
    return model.ObjVal
    

def deterministic_Stackelberg_menu(m, n, k, mu, A, B):
    """
    m: num of leader actions
    n: number of follower actions
    k: number of agent types
    mu: prior distribution over agent types
    A: shape of [k, m, n], leader payoff matrix
    B: shape of [k, m, n], follower payoff matrix
    """

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()

    agent_action = [np.arange(n)]
    agent_actions = cartesian_product_transpose(*(agent_action*k))
    res = 0.0
    for action_index, action in enumerate(agent_actions):
        model = gp.Model('deterministic_Stackelberg_menu_{}'.format(action_index), env=env)
        X = []
        for theta in range(k):
            x_theta = []
            for i in range(m):
                x_theta += [model.addVar(lb=0.0, ub=1.0, name='x_{}_{}'.format(theta, i))]
            model.update()
            model.addConstr(gp.quicksum(x_theta) == 1.0, name="x_{}_sum".format(theta))
            X.append(x_theta)
        
        for theta in range(k):
            theta_utility = 0.0
            for i in range(m):
                theta_utility += B[theta][i][action[theta]] * X[theta][i]
            
            for theta_prime in range(k):
                for j_prime in range(n):
                    theta_prime_j_prime_utility = 0.0
                    for i in range(m):
                        theta_prime_j_prime_utility += B[theta][i][j_prime] * X[theta_prime][i]
                    model.addConstr(theta_utility >= theta_prime_j_prime_utility, name='ic_{}_{}_{}'.format(theta, theta_prime, j_prime))
        
        objective = 0.0
        for theta in range(k):
            theta_obj = 0.0
            for i in range(m):
                theta_obj += X[theta][i] * A[theta][i][action[theta]]
            theta_obj *= mu[theta]
            objective += theta_obj
        model.setObjective(objective, GRB.MAXIMIZE)

        model.optimize()
        if model.Status == 2: res = max(res, model.ObjVal)
    return res

    
