#!/usr/bin/env python3.7

# Copyright 2022, Minbiao Han

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
count = 0
greater_dynamic = 0
count2 = 0
bug = 0
greater_SM, bug2, bug3 = 0, 0, 0
greater, equal, small = 0, 0, 0
logging = open("results/logging_2x2.txt", "w")
logging.write("\n\n\n\n\n\n")
logging.write(str(time.asctime(time.localtime(time.time()))))
logging.write("\n")

def addConstr_second_round(x, j, k, k_prime, B, model):
    model.addConstr(x[j, 0]*B[0][k] + x[j, 1]*B[1][k] >= x[j, 0] *B[0][k_prime] + x[j, 1]*B[1][k_prime])


def addConstr_first_round(x1, x2, j, j_prime, k, k_prime, B, model):
    model.addConstr(x1[0]*B[0][j] + x1[1] * B[1][j] + x2[j, 0]*B[0][k] + x2[j,1] *B[1][k] >= x1[0]*B[0][j_prime] + x1[1] * B[1][j_prime] + x2[j_prime, 0]*B[0][k_prime] + x2[j_prime,1] *B[1][k_prime])


for i in range(1000):
    pi=0.5
    A1 = np.random.rand(2, 2)
    A2 = A1.copy()
    B1 = np.random.rand(2, 2)
    B2 = np.random.rand(2, 2)

    BSE_utility = -float("inf")
    BSE_leader_strategy = [0.0, 0.0]
    for j1 in range(2):
        for j2 in range(2):
            model = gp.Model("BSE")
            x = model.addVars(2, lb=0.0,ub=1.0, name="leader_strategy")
            for j1_prime in range(2):
                if j1_prime == j1: continue
                model.addConstr(x[0]*B1[0][j1] + x[1]*B1[1][j1] >= x[0]*B1[0][j1_prime] + x[1]*B1[1][j1_prime])
            for j2_prime in range(2):
                if j2_prime == j2: continue
                model.addConstr(x[0]*B2[0][j2] + x[1]*B2[1][j2] >= x[0]*B2[0][j2_prime] + x[1]*B2[1][j2_prime])
            model.addConstr(x[0]+x[1] == 1.0)
            model.setObjective((pi*(x[0]*A1[0][j1]+x[1]*A1[1][j1]) + (1-pi)*(x[0]*A2[0][j2]+x[1]*A2[1][j2])), GRB.MAXIMIZE)
            try:
                model.optimize()
            except gp.GurobiError:
                print("Optimization Error")
            # model status: 3 - infeasible; 2 - optimal
            if model.Status == 2:
                if model.ObjVal > BSE_utility:
                    BSE_utility = max(BSE_utility, model.ObjVal)
                    for i in range(2):
                        BSE_leader_strategy[i] = float(format(x[i].x, '.6f'))
            else: print("infeasible model")

    SM_utility = - float("inf")
    SM_leader_strategy1 = np.array([0.0, 0.0])
    SM_leader_strategy2 = np.array([0.0, 0.0])
    SM_follower_j1 = -1
    SM_follower_j2 = -1
    for j1 in range(2):
        for j2 in range(2):
            model = gp.Model("BSE2")
            x1 = model.addVars(2, lb=0.0, ub=1.0, name="leader_strategy")
            x2 = model.addVars(2, lb=0.0, ub=1.0, name="leader_strategy")
            for j1_prime in range(2):
                model.addConstr(
                    x1[0] * B1[0][j1] + x1[1] * B1[1][j1] >= x2[0] * B1[0][j1_prime] + x2[1] * B1[1][j1_prime])
                if j1_prime == j1: continue
                model.addConstr(
                    x1[0] * B1[0][j1] + x1[1] * B1[1][j1] >= x1[0] * B1[0][j1_prime] + x1[1] * B1[1][j1_prime])
            for j2_prime in range(2):
                model.addConstr(
                    x2[0] * B2[0][j2] + x2[1] * B2[1][j2] >= x1[0] * B2[0][j2_prime] + x1[1] * B2[1][j2_prime])
                if j2_prime == j2: continue
                model.addConstr(
                    x2[0] * B2[0][j2] + x2[1] * B2[1][j2] >= x2[0] * B2[0][j2_prime] + x2[1] * B2[1][j2_prime])
            model.addConstr(x1[0] + x1[1] == 1.0)
            model.addConstr(x2[0] + x2[1] == 1.0)
            model.setObjective(pi * (x1[0] * A1[0][j1] + x2[0] * A2[0][j2] + x1[1] * A1[1][j1] + x2[1] * A2[1][j2]),
                               GRB.MAXIMIZE)
            try:
                model.optimize()
            except gp.GurobiError:
                print("Optimization Error")
            # model status: 3 - infeasible; 2 - optimal
            if model.Status == 2:
                if model.ObjVal > SM_utility:
                    SM_utility = max(SM_utility, model.ObjVal)
                    for i in range(2):
                        SM_leader_strategy1[i] = float(format(x1[i].x, '.6f'))
                        SM_leader_strategy2[i] = float(format(x2[i].x, '.6f'))
                    SM_follower_j1, SM_follower_j2 = j1, j2
            else:
                print("infeasible model")

    dynamic_utility = -float("inf")
    dynamic_strategy_1 = np.array([0.0, 0.0])
    dynamic_strategy_2 = np.zeros((2, 2))
    dynamic_follower_j1, dynamic_follower_j2 = -1, -1
    dynamic_follower_k1, dynamic_follower_k2 = -1, -1
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    model = gp.Model("dynamic_true")
                    x1 = model.addVars(2, lb=0.0, ub=1.0, name="dynamic_strategy_first_round")
                    x2 = model.addVars(2, 2, lb=0.0, ub=1.0, name="dynamic_strategy_second_round")
                    for k1_prime in range(2):
                        if k1_prime == k1: continue
                        addConstr_second_round(x2, j1, k1, k1_prime, B1, model)
                    for k2_prime in range(2):
                        if k2_prime == k2: continue
                        addConstr_second_round(x2, j2, k2, k2_prime, B2, model)
                    for j1_prime in range(2):
                        for k1_prime in range(2):
                            addConstr_first_round(x1, x2, j1, j1_prime, k1, k1_prime, B1, model)
                    for j2_prime in range(2):
                        for k2_prime in range(2):
                            addConstr_first_round(x1, x2, j2, j2_prime, k2, k2_prime, B2, model)
                    model.addConstr(x1[0] + x1[1] == 1.0)
                    for j_prime in range(2):
                        model.addConstr(x2[j_prime, 0] + x2[j_prime, 1] == 1.0)
                    model.setObjective(pi*(x1[0]*A1[0][j1]+x1[1]*A1[1][j1]) + pi*(x1[0]*A2[0][j2]+x1[1]*A2[1][j2]) + pi*(x2[j1, 0]*A1[0][k1] +x2[j1,1]*A1[1][k1]) + pi*(x2[j2, 0]*A2[0][k2] + x2[j2, 1]*A2[1][k2]), GRB.MAXIMIZE)
                    try:
                        model.optimize()
                    except gp.GurobiError:
                        print("Optimization Error")
                    if model.Status == 2:
                        if model.ObjVal > dynamic_utility:
                            dynamic_utility = max(dynamic_utility, model.ObjVal)
                            dynamic_follower_j1, dynamic_follower_j2 = j1, j2
                            dynamic_follower_k1, dynamic_follower_k2 = k1, k2
                            for a in range(2):
                                dynamic_strategy_1[a] = float(format(x1[a].x, '.6f'))
                            for a in range(2):
                                for b in range(2):
                                    dynamic_strategy_2[a, b] = float(format(x2[a, b].x, '.6f'))
                    else:
                        print("infeasible model")

    print("*" * 100)
    BSE_utility = BSE_utility * 2.0
    SM_utility = SM_utility * 2.0
    print("BSE utility: ", BSE_utility, "SM utility: ", SM_utility, "dySE utility: ", dynamic_utility)
    logging.write("BSE utility: " + str(BSE_utility)  + ", SM utility: " + str(SM_utility) + ", dynamic utility: " + str(dynamic_utility) + "\n")

    if abs(SM_utility - dynamic_utility) > 0.00001:
        if SM_utility < dynamic_utility:
            greater_SM += 1
        else:
            greater_dynamic += 1
    if abs(SM_utility - BSE_utility) > 0.00001:
        if SM_utility < BSE_utility:
            bug += 1
        else:
            count += 1
    if abs(dynamic_utility - BSE_utility) > 0.00001:
        if dynamic_utility < BSE_utility:
            bug2 += 1
        else:
            count2 += 1
    print("count: ",count, "bug:", bug, "greater_dynamic :", greater_dynamic,"greater_SM:", greater_SM, "count2: ", count2,"bug2:",bug2, "*"*100)
    logging.write("count: " + str(count) + " bug " + str(bug) + "greater_dynamic :" + str(greater_dynamic) + " greater_SM " + str(greater_SM) + "count2: " + str(count2) + " bug2 " + str(bug2) + "\n")
    logging.write("BSE Leader strategy: " + str(BSE_leader_strategy) + "\n")
    logging.write("BSE Leader menu: " + str(SM_leader_strategy1) + str(SM_leader_strategy2) + "BSE Follwer: " + str(SM_follower_j1) + str(SM_follower_j2) + "\n")
    logging.write("dynamic Strategy 1: " + str(dynamic_strategy_1)+"\n" +"dynamic strategy 2: " + str(dynamic_strategy_2[0,:]) + " " + str(dynamic_strategy_2[1,:]) + "\n" +str(dynamic_follower_j1) + "   "+str(dynamic_follower_j2) + "   " +str(dynamic_follower_k1)+ "   " +str(dynamic_follower_k2)+"\n")
    logging.write("^"*50+"\n")
    logging.write(str(A1) + "\n")
    logging.write(str(A2) + "\n")
    logging.write(str(B1) + "\n")
    logging.write(str(B2) + "\n\n")
logging.write("count: "+str(count)+" bug " + str(bug)+ "greater_dynamic :"+ str(greater_dynamic)+ " greater_SM " +str(greater_SM)+"count2: "+ str(count2)+" bug2 " +str(bug2)+"\n")
logging.close()
print("count: ",count, "bug:", bug, "greater_dynamic :", greater_dynamic,"greater_SM:", greater_SM, "count2: ", count2,"bug2:",bug2, "*"*100)
