import gurobipy as gp
import itertools

def get_history_str(history, n, T):
    history_str = ""
    for t in range(T):
        history_str += "{}-".format(history % n)
        history = history // n
    return history_str[:-1]

"""
    Solves for the dySE for the repeated bayesian stackelberg game given by R, C, pi, and T
        R:  mxn payoff matrix for the leader
        C:  list of payoff matrices, one for each follower type
        pi: prior distribution for follower types
        T:  number of rounds
"""
def dySE(R, C, pi, T, security=True):
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    ntypes = len(C)
    if not security:
        R = [R for _ in range(ntypes)]

    m = len(R[0])
    n = len(R[0][0])

    bestObj = -1000000
    bestModel = None

    for fullhistories in itertools.product(range(n**T), repeat=ntypes):
        M = gp.Model("dySE", env=env)

        # Create variables: leader strategy X, follower strategy y, and follower utilities a_theta

        #   x_j,i^t = X[Time t][Follower history (bold) j][Action i]
        X = []

        for t in range(T):
            xt = []
            nv = n**t
            for history in range(nv):
                xt_hist = []
                for i in range(m):
                    xt_hist += [M.addVar(lb=0.0, ub=1.0, name="x{}_{},{}".format(t, i, get_history_str(history, n, t)))]
                M.update()
                M.addConstr(gp.quicksum(xt_hist) == 1.0, name="ph_{},{}".format(nv, i))
                xt.append(xt_hist)

            X.append(xt)

        # Create expression for objective function
        objective = 0

        for t in range(T):
            for theta in range(ntypes):
                nv = n**(T - t - 1)
                history = fullhistories[theta] // nv
                j = history % n
                history = history // n
                for i in range(m):
                    objective += pi[theta]*R[theta][i][j]*X[t][history][i]
        M.setObjective(objective, gp.GRB.MAXIMIZE)

        # Add IC constraints for follower

        M_c = float(2**10) # Large constant assumed to be larger than any achievable utility

        for theta in range(ntypes):
            follower_utility = 0
            subhistory = fullhistories[theta]
            for t in range(T):
                j = subhistory % n
                subhistory = subhistory // n
                for i in range(m):
                    follower_utility += C[theta][i][j] * X[T-t-1][subhistory][i]

            for alternatehistory in range(n**T):
                alternate_utility = 0
                subhistory = alternatehistory
                for t in range(T):
                    j = subhistory % n
                    subhistory = subhistory // n
                    for i in range(m):
                        alternate_utility += C[theta][i][j] * X[T-t-1][subhistory][i]

                M.addConstr(follower_utility >= alternate_utility, name="lb{}_{}".format(theta, get_history_str(alternatehistory, n, T)))

        M.optimize()
        if not M.status == gp.GRB.INFEASIBLE and M.objVal >= bestObj:
            M.update()
            bestObj = M.objVal
            bestModel = M.copy()
        M.dispose()
    bestModel.optimize()
    return bestModel

"""
    Solves for the dySE for the repeated bayesian stackelberg game given by R, C, pi, and T
        R:  mxn payoff matrix for the leader
        C:  list of payoff matrices, one for each follower type
        pi: prior distribution for follower types
        T:  number of rounds
"""
def dySE_binary(R, C, pi, T, security=True):
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    M = gp.Model("dySE", env=env)

    ntypes = len(C)
    if not security:
        R = [R for _ in range(ntypes)]

    m = len(R[0])
    n = len(R[0][0])

    # Create variables: leader strategy X, follower strategy y, and follower utilities a_theta

    #   x_j,i^t = X[Time t][Follower history (bold) j][Action i]
    X = []

    for t in range(T):
        xt = []
        nv = n**t
        for history in range(nv):
            xt_hist = []
            for i in range(m):
                xt_hist += [M.addVar(lb=0.0, ub=1.0, name="x{}_{},{}".format(t, i, get_history_str(history, n, t)))]
            M.update()
            M.addConstr(gp.quicksum(xt_hist) == 1.0, name="ph_{},{}".format(nv, i))
            xt.append(xt_hist)

        X.append(xt)

    #   y^theta_t,j = y[Time t][Type theta][Action j]
    y = []

    for theta in range(ntypes):
        y_theta = []
        for fullhistory in range(n**T):
            y_theta += [M.addVar(vtype=gp.GRB.BINARY, name="y{}_{}".format(theta, get_history_str(fullhistory, n, T)))]
        M.update()
        M.addConstr(gp.quicksum(y_theta) == 1.0, name="sa{}".format(theta))
        y.append(y_theta)

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
                for i in range(m):
                    for j in range(n):
                        nve = n**(T-t-1)
                        for extension in range(nve):
                            fullhistory = (n*history+j)*nve + extension
                            objective += pi[theta]*R[theta][i][j]*X[t][history][i]*y[theta][fullhistory]
    M.setObjective(objective, gp.GRB.MAXIMIZE)

    # Add IC constraints for follower

    M_c = float(2**10) # Large constant assumed to be larger than any achievable utility

    for fullhistory in range(n**T):
        for theta in range(ntypes):
            follower_utility = 0
            subhistory = fullhistory
            for t in range(T):
                j = subhistory % n
                subhistory = subhistory // n
                for i in range(m):
                    follower_utility += C[theta][i][j] * X[T-t-1][subhistory][i]

            M.addConstr(A[theta] - follower_utility >= 0, name="lb{}_{}".format(theta, get_history_str(fullhistory, n, T)))
            M.addConstr(A[theta] - follower_utility <= M_c*(1 - y[theta][fullhistory]), name="ub{}_{}".format(theta, get_history_str(fullhistory, n, T)))

    M.optimize()



    # logging_X = []
    # logging_y = []
    # for t in range(T):
    #     logging_xt = []
    #     logging_nv = n**t
    #     for logging_history in range(logging_nv):
    #         logging_xt_hist = []
    #         for i in range(m):
    #             logging_xt_hist += [X[t][logging_history][i].x]
    #         logging_xt.append(logging_xt_hist)

    #     logging_X.append(logging_xt)
    
    # for theta in range(ntypes):
    #     logging_y_theta = []
    #     for fullhistory in range(n**T):
    #         logging_y_theta += [y[theta][fullhistory].x]
    #         #y_theta += [model.addVar(vtype=gp.GRB.BINARY, name="y{}_{}".format(theta, fullhistory))]
    #     #model.update()
    #     #model.addConstr(gp.quicksum(y_theta) == 1.0, name="sa{}".format(theta))
    #     logging_y.append(logging_y_theta)


    return M#, logging_X, logging_y

"""
    Attempt to extend the reward from final round of the binary solution
"""
def dySE_extended_utility(R, C, pi, T, security=True):
    ntypes = len(C)
    if ntypes >= T-1:
        return dySE_binary(R, C, pi, T, security).objVal

    M = dySE_binary(R, C, pi, ntypes + 1, security)

    ntypes = len(C)
    if not security:
        R = [R for _ in range(ntypes)]

    m = len(R[0])
    n = len(R[0][0])

    extra_rounds = T - ntypes - 1
    objective = M.objVal

    for theta in range(ntypes):
        nv = n**ntypes
        for history in range(nv):
            for i in range(m):
                for j in range(n):
                    fullhistory = n*history + j
                    objective += extra_rounds*pi[theta]*R[theta][i][j] * M.getVarByName("x{}_{},{}".format(ntypes, i, history)).X * M.getVarByName("y{}_{}".format(theta, fullhistory)).X

    return objective


"""
    Solves for the dySE for the repeated bayesian stackelberg game given by R, C, pi, and T
        R:  mxn payoff matrix for the leader
        C:  list of payoff matrices, one for each follower type
        pi: prior distribution for follower types
        T:  number of rounds
"""
def dySE_fast(R, C, pi, T, security=True):
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    M = gp.Model("dySE", env=env)
    M.Params.NonConvex = 2

    ntypes = len(C)
    if not security:
        R = [R for _ in range(ntypes)]

    m = len(R[0])
    n = len(R[0][0])

    # Create variables: leader strategy X, follower strategy y, and follower utilities a_theta

    #   x_j,i^t = X[Time t][Follower history (bold) j][Action i]
    X = []

    for t in range(T):
        xt = []
        nv = n**t
        for history in range(nv):
            xt_hist = []
            for i in range(m):
                xt_hist += [M.addVar(lb=0.0, ub=1.0, name="x{}_{},{}".format(t, i, get_history_str(history, n, t)))]
            M.update()
            M.addConstr(gp.quicksum(xt_hist) == 1.0, name="ph_{},{}".format(t, get_history_str(history, n, t)))
            xt.append(xt_hist)

        X.append(xt)

    #   y^theta_t,j = y[Type theta][Time t][Action j]
    y = []

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
                for i in range(m):
                    for j in range(n):
                        objective += pi[theta]*R[theta][i][j]*X[t][history][i]*z[theta][t][history*n + j]
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
                    follower_utility += C[theta][i][j] * X[T-t-1][subhistory][i]

            M.addConstr(A[theta] - follower_utility >= 0, name="lb{}_{}".format(theta, get_history_str(fullhistory, n, T)))
            M.addConstr(A[theta] - follower_utility <= M_c*(T - gp.quicksum(y_thetas)), name="ub{}_{}".format(theta, get_history_str(fullhistory, n, T)))

    M.optimize()
    return M

"""
    Solves for the dySE for the repeated bayesian stackelberg game given by R, C, pi, and T
        R:  mxn payoff matrix for the leader
        C:  list of payoff matrices, one for each follower type
        pi: prior distribution for follower types
        T:  number of rounds
"""
def dySE_MILP(R, C, pi, T, security=True):
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    M = gp.Model("dySE", env=env)
    M.Params.NonConvex = 2

    ntypes = len(C)
    if not security:
        R = [R for _ in range(ntypes)]

    m = len(R[0])
    n = len(R[0][0])

    # Create variables: leader strategy X, follower strategy y, and follower utilities a_theta

    #   x_j,i^t = X[Time t][Follower history (bold) j][Action i]
    X = []

    for t in range(T):
        xt = []
        nv = n**t
        for history in range(nv):
            xt_hist = []
            for i in range(m):
                xt_hist += [M.addVar(lb=0.0, ub=1.0, name="x{}_{},{}".format(t, i, get_history_str(history, n, t)))]
            M.update()
            M.addConstr(gp.quicksum(xt_hist) == 1.0, name="ph_{},{}".format(t, get_history_str(history, n, t)))
            xt.append(xt_hist)

        X.append(xt)

    #   y^theta_t,j = y[Type theta][Time t][Action j]
    y = []

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
    objective = 0.0

    for t in range(T):
        for theta in range(ntypes):
            nv = n**t
            for history in range(nv):
                for i in range(m):
                    for j in range(n):
                        zt_theta_hist = z[theta][t][history*n + j]
                        xt_hist = X[t][history][i]
                        wt_theta_hist = M.addVar(name="w{}_{},{};{}".format(theta, t, get_history_str(history, n, t), i))
                        M.update()
                        M.addConstr(wt_theta_hist <= zt_theta_hist, name="leqwz_{},{};{}".format(theta, t, get_history_str(history, n, t), i))
                        M.addConstr(wt_theta_hist <= xt_hist, name="leqwx_{},{};{}".format(theta, t, get_history_str(history, n, t), i))
                        objective += pi[theta]*R[theta][i][j]*wt_theta_hist
    M.update()
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
                    follower_utility += C[theta][i][j] * X[T-t-1][subhistory][i]

            M.addConstr(A[theta] - follower_utility >= 0, name="lb{}_{}".format(theta, get_history_str(fullhistory, n, T)))
            M.addConstr(A[theta] - follower_utility <= M_c*(T - gp.quicksum(y_thetas)), name="ub{}_{}".format(theta, get_history_str(fullhistory, n, T)))

    M.optimize()
    return M
