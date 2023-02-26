import gurobipy as gp
from gurobipy import GRB
import numpy as np

"""
    Solves for the SSE leader strategy for the stackelberg game given by m, n, A, B
        m: number of leader actions
        n: number of follower actions
        A: leader payoff matrix
        B:  follower payoff matrix
"""
def StackelbergEquilibrium(m, n, A, B):
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    res = 0.0
    opt_vars = None
    for action in range(n):
        model = gp.Model('Stackelberg_strategy_{}'.format(action), env=env)
        X = []
        for i in range(m):
            X += [model.addVar(lb=0.0, ub=1.0, name='x_{}'.format(i))]
        model.update()
        model.addConstr(gp.quicksum(X) == 1.0, name="x_sum")
        theta_utility = 0.0
        for i in range(m):
            theta_utility += B[i][action] * X[i]
        
        for j_prime in range(n):
            theta_prime_j_prime_utility = 0.0
            for i in range(m):
                theta_prime_j_prime_utility += B[i][j_prime] * X[i]
            model.addConstr(theta_utility >= theta_prime_j_prime_utility, name='ic_{}'.format(j_prime))
        objective = 0.0
        for i in range(m):
            objective += X[i] * A[i][action]
        model.setObjective(objective, GRB.MAXIMIZE)
        model.optimize()
        if model.Status == 2:
            if model.ObjVal > res:
                res = model.ObjVal
                opt_vars = []
                for v in model.getVars():
                    opt_vars += [(v.varName, v.x)]
    return res, opt_vars

"""
    Solves for the SSE leader strategy for the stackelberg game given by m, n, A, B, mu with perfect information
        m: number of leader actions
        n: number of follower actions
        k: number of agent types
        mu: prior distribution over agent types
        A: shape of [k, m, n], leader payoff matrix
        B: shape of [k, m, n], follower payoff matrix
"""
def PerfectLearning(m, n, k, mu, A, B):
    res = 0.0
    opt_vars_list = []
    for A_i, B_i, mu_i in zip(A, B, mu):
        utility, opt_vars = StackelbergEquilibrium(m, n, A_i, B_i)
        res += mu_i*utility
        opt_vars_list += [opt_vars]
    return res, opt_vars_list
