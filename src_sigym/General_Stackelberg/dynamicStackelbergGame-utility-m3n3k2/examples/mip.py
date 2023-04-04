from gurobipy import *

if __name__ == '__main__':
    #Create a new model
    m = Model("Investment");
    #Create variables, delete lb and ub and change vtype=GRB.BINARY if we are creating binary integer variables
    x1 = m.addVar(lb=1.0, ub=10.0, vtype=GRB.INTEGER, name="x1")
    x2 = m.addVar(lb=1.0, ub=10.0, vtype=GRB.INTEGER, name="x2")
    x3 = m.addVar(lb=1.0, ub=10.0, vtype=GRB.INTEGER, name="x3")
    x4 = m.addVar(lb=1.0, ub=10.0, vtype=GRB.INTEGER, name="x4")
    x5 = m.addVar(lb=1.0, ub=10.0, vtype=GRB.INTEGER, name="x5")

    #Intigrate new variables
    m.update()

    #Set Objective
    m.setObjective(160*x1 + 160*x2 + 160*x3 + 75*x4 + 75*x5, GRB.MINIMIZE)
    m.addConstr(   x1 +    x2 +    x3                   >= 3,  "c0")
    m.addConstr(   x1                                   >= 1,  "c1")
    m.addConstr(           x2                           >= 0,  "c2")
    m.addConstr(                   x3                   >= 1,  "c3")
    m.addConstr(                           x4           >= 0,  "c4")
    m.addConstr(                                   x5   >= 0,  "c5")
    m.addConstr(40*x1 + 40*x2 + 40*x3 + 25*x4 + 25*x5   >= 365,"c6")

    m.optimize()

    for v in m.getVars():
        print(v.varName, v.x)
    print("Obj:", m.objVal)
