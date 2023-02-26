import gurobipy as gp


"""
    Solves for the single round BSE for the repeated bayesian stackelberg game given by R, C, pi, and T
        R:  mxn payoff matrix for the leader
        C:  list of payoff matrices, one for each follower type
        pi: prior distribution for follower types
        T:  number of rounds
"""
def BSE_binary(mu, r, P, c):
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    M = gp.Model("BSE", env=env)

    ntypes = len(mu)
    
    m = len(r)
    n = len(c[0])

    assert len(P) == ntypes
    assert len(P[0]) == n
    assert len(P[0][0]) == m

    # Create variables: leader contract X, follower strategy y, and follower utilities a_theta

    #   x_i = X[Action i]
    X = []

    for i in range(m):
        X += [M.addVar(lb=0.0, ub=1.0, name="x{}".format(i))]
    M.update()
    
    #   y^theta_t,j = y[Type theta][Action j]
    y = []

    for theta in range(ntypes):
        y_theta = []
        for j in range(n):
            y_theta += [M.addVar(vtype=gp.GRB.BINARY, name="y{}_{}".format(theta, j))]
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

    for theta in range(ntypes):
        for i in range(m):
            for j in range(n):
                objective += mu[theta]*P[theta][j][i]*(r[i] - X[i])*y[theta][j]
    
    M.setObjective(objective, gp.GRB.MAXIMIZE)

    # Add IC constraints for follower

    M_c = float(2**10) # Large constant assumed to be larger than any achievable utility

    for j in range(n):
        for theta in range(ntypes):
            follower_utility = 0
            for i in range(m):
                follower_utility += C[theta][i][j] * X[i]

            M.addConstr(A[theta] - follower_utility >= 0, name="lb{}_{}".format(theta, j))
            M.addConstr(A[theta] - follower_utility <= M_c*(1 - y[theta][j]), name="ub{}_{}".format(theta, j))

    M.optimize()
    return M
