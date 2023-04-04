import gurobipy as gp
import itertools
import numpy as np



def get_history_str(history, n, T):
    history_str = ""
    for t in range(T):
        history_str += "{}-".format(history % n)
        history = history // n
    return history_str[:-1]



"""
    Solves for the dynamic contract for the repeated bayesian Contract Design given by mu, r, P, c, and T
        mu: prior distribution over agent types
        r: shape of [m], rewards of each outcome
        P: shape of [k, n, m], probabilistic distribution over outcomes of all agent types
        c: shape of [k, n], cost over every action of all agent types
        T: number of interaction rounds
"""
def dyCo(mu, r, P, c, T):

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    M = gp.Model("dyCo", env=env)
    M.Params.NonConvex = 2

    ntypes = len(mu)
    
    m = len(r)
    n = len(c[0])

    assert len(P) == ntypes
    assert len(P[0]) == n
    assert len(P[0][0]) == m

    # Create variables: leader strategy X, follower strategy y, and follower utilities a_theta

    #   x_j,i^t = X[Time t][Follower history (bold) j_{t-1}][Action i]
    X = []

    loggingX = []

    for t in range(T):
        xt = []
        nv = n**t
        for history in range(nv):
            xt_hist = []
            for i in range(m):
                xt_hist += [M.addVar(lb=0.0, name="x{},{},{}".format(t, get_history_str(history, n, t), i))]
            M.update()
            #M.addConstr(gp.quicksum(xt_hist) == 1.0, name="ph_{},{}".format(t, get_history_str(history, n, t)))
            xt.append(xt_hist)

        X.append(xt)

    #   y^theta_t,j = y[Type theta][Time t][Action j]
    y = []

    loggingy = []

    for theta in range(ntypes):
        y_theta = []
        for t in range(T):
            yt_theta = []
            for j in range(n):
                yt_theta += [M.addVar(vtype=gp.GRB.BINARY, name="y{}_{}_{}".format(theta, t, j))]
            M.update()
            M.addConstr(gp.quicksum(yt_theta) == 1.0, name="sa{}".format(theta))
            y_theta.append(yt_theta)
        y.append(y_theta)

    #   z^theta,t_j = z[Type theta][Time t][Follower history (bold) j]
    z = []

    loggingz = []

    for theta in range(ntypes):
        z_theta = []
        for t in range(T):
            zt_theta = []
            nv = n**(t+1)
            for history in range(nv):
                zt_theta_hist = M.addVar(lb=0.0, ub=1.0, name="z{}_{},{}".format(theta, t+1, get_history_str(history, n, t+1)))
                M.update()
                jt = history % n
                prev_history = history // n
                y_curr = y[theta][t][jt]
                if t > 0:
                    z_prev = z_theta[t-1][prev_history]
                    M.addConstr(zt_theta_hist <= y_curr, name="leqandy_{},{},{}".format(theta, t, get_history_str(history, n, t+1), jt))
                    M.addConstr(zt_theta_hist <= z_prev, name="leqandz_{},{},{}".format(theta, t, get_history_str(history, n, t+1), jt))
                    M.addConstr(zt_theta_hist >= y_curr + z_prev - 1, name="geqand_{},{},{}".format(theta, t, get_history_str(history, n, t+1), jt))
                else:
                    M.addConstr(zt_theta_hist == y_curr, name="eq_{},{},{},{}".format(theta, t, get_history_str(history, n, t+1), jt))
                zt_theta.append(zt_theta_hist)
            z_theta.append(zt_theta)
        z.append(z_theta)

    #   a_theta = A[Type theta]
    A = []

    for theta in range(ntypes):
        A += [M.addVar(name="a{}".format(theta))]
    M.update()

    # Create expression for objective function
    objective = 0

    for t in range(T):
        for theta in range(ntypes):
            nv = n**t
            for history in range(nv):
                for jt in range(n):
                    for i in range(m):
                        objective += mu[theta]*P[theta][jt][i]*z[theta][t][history*n + jt]*(r[i] - X[t][history][i])
    M.setObjective(objective, gp.GRB.MAXIMIZE)

    # Add IC constraints for follower

    M_c = float(2**10) # Large constant assumed to be larger than any achievable utility

    for fullhistory in range(n**T):
        for theta in range(ntypes):
            follower_utility = 0
            subhistory = fullhistory
            y_thetas = []
            for t in range(T):
                j = subhistory % n
                y_thetas.append(y[theta][T-t-1][j])
                subhistory = subhistory // n
                for i in range(m):
                    follower_utility += P[theta][j][i] * X[T-t-1][subhistory][i]
                follower_utility -= c[theta][j]

            M.addConstr(A[theta] - follower_utility >= 0, name="lb{}_{}".format(theta, get_history_str(fullhistory, n, T)))
            M.addConstr(A[theta] - follower_utility <= M_c*(T - gp.quicksum(y_thetas)), name="ub{}_{}".format(theta, get_history_str(fullhistory, n, T)))

    M.optimize()

    for t in range(T):
        xt = []
        nv = n**t
        for history in range(nv):
            xt_hist = []
            for i in range(m):
                xt_hist += [X[t][history][i].x]
            xt.append(xt_hist)
        loggingX.append(xt)
    
    for theta in range(ntypes):
        y_theta = []
        for t in range(T):
            yt_theta = []
            for j in range(n):
                yt_theta += [y[theta][t][j].x]
            y_theta.append(yt_theta)
        loggingy.append(y_theta)
    
    for theta in range(ntypes):
        z_theta = []
        for t in range(T):
            zt_theta = []
            nv = n**(t+1)
            for history in range(nv):
                zt_theta.append(z[theta][t][history].x)
            z_theta.append(zt_theta)
        loggingz.append(z_theta)

    return M, loggingX, loggingy, loggingz

if __name__=='__main__':
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
    mu = np.array([0.5, 0.5])
    model, _, _, _ = dyCo(mu, r, P, c, T=3)
    print(model.objVal, "ttttttt")