import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from General_Stackelberg.dynamic_stackelberg import dyse, SSE
from General_Stackelberg.dynamic_stackelberg import utils
import utils
from sigym import Follower, Platform
import tkinter

# def test_without_follower_class():
#     defender_mode_dict = {
#         '1': "random",
#         '2': 'MWU'
#     }
#     attacker_mode_dict = {
#         '1': "random",
#         '2': 'BestResponse'
#     }
#     defender_mode = input("Please select defender behavior mode: \n1 for random; 2 for MWU: \n")
#     print("you selected " + defender_mode_dict[defender_mode] + " defender.")
#     attacker_mode = input("Please select attacker behavior mode: \n1 for random; 2 for BestResponse: \n")
#     print("you selected " + attacker_mode_dict[attacker_mode] + " attacker.")
#     str_T = input("Please enter the number of interaction rounds: ")
#     print("The total number of interaction rounds is " + str_T + ".")
#     T = int(str_T)
#     str_trial = input("Please enter the number of trials: ")
#     print("The total number of interaction rounds is " + str_trial + ".")
#     trial = int(str_trial)
#     m, n = 3, 3
#     eps = 0.01

#     if defender_mode == '2':
#         rgt = 0.0
#         for tr in range(trial):
#             R = np.loadtxt("random_instance/R_{}.txt".format(tr))
#             C = np.loadtxt("random_instance/C_{}.txt".format(tr))
#             u_sse, _ = SSE.StackelbergEquilibrium(2, 2, R, C)
#             x = [1.0/m for _ in range(m)]
#             u_t = 0.0
#             for t in range(T):
#                 j_t = None
#                 if attacker_mode == '1':
#                     j_t = np.random.choice(np.arange(n))
#                 else:
#                     j_t = utils.best_response(x, C)
#                 i_t = np.random.choice(np.arange(m), p=x)
#                 u_t += R[i_t][j_t]
#                 print("At round " + str(t) + ", the leader plays action " + str(i_t) + " according to mixed strategy " + str(x) + ", the follower responses by " + str(j_t) + ".")
#                 print("The leader utility at current round is {}, and the cumulative leader utility until current round is {}.".format(R[i_t][j_t], u_t))
#                 print("==============================="*5)
#                 reward_t = R[:, j_t]
#                 x = utils.mwu_update(x=x, reward=reward_t, eps=eps)
#             rgt += (u_sse*T - u_t)/T
#         print("The averaged regret you get over {} trials is {}".format(trial, rgt/trial))
    
#     else:
#         rgt = 0.0
#         for tr in range(trial):
#             R = np.loadtxt("random_instance/R_{}.txt".format(tr))
#             C = np.loadtxt("random_instance/C_{}.txt".format(tr))
#             u_sse, _ = SSE.StackelbergEquilibrium(2, 2, R, C)
#             x = [np.random.rand() for i in range(m)]
#             temp = sum(x)
#             x = [i/temp for i in x]
#             u_t = 0.0
#             for t in range(T):
#                 j_t = None
#                 if attacker_mode == '1':
#                     j_t = np.random.choice(np.arange(n))
#                 else:
#                     j_t = utils.best_response(x, C)
#                 i_t = np.random.choice(np.arange(m), p=x)
#                 u_t += R[i_t][j_t]
#                 print("At round " + str(t) + ", the leader plays action " + str(i_t) + " according to mixed strategy " + str(x) + ", the follower responses by " + str(j_t) + ".")
#                 print("The leader utility at current round is {}, and the cumulative leader utility until current round is {}.".format(R[i_t][j_t], u_t))
#                 print("==============================="*5)
#                 x = [np.random.rand() for i in range(m)]
#                 temp = sum(x)
#                 x = [i/temp for i in x]
#             rgt += (u_sse*T - u_t)/T
#         print("The averaged regret you get over {} trials is {}".format(trial, rgt/trial))

defender_mode_dict = {
    '1': "random",
    '2': 'MWU'
}

attacker_mode_dict = {
    '1': "random",
    '2': 'best_response'
}

defender_mode = '1'
attacker_mode = '2'
T = 10
m, n = 3, 3 #m defender row / n attacker colomn
eps = 0.01
trial = 2
imgheight = 800
imgwidth = 800
base = 50, 50
base2 = imgwidth/2+50, 50
squareL = min((imgwidth - base[0]*2) / m, (imgheight - base[1]*2) / n )
i_res = []
j_res = []
update_cnt = 0


# init tk
root = tkinter.Tk()
myCanvas = tkinter.Canvas(root, bg="white", height=imgheight, width=imgwidth)
string_var = tkinter.StringVar()
caption = tkinter.Label(root, textvariable = string_var, font=("Arial", 25))
caption.place(relx = 0.5,
                rely = 0.9,
                anchor ='center')

'''
def update_win():
    global update_cnt, root
    if update_cnt == len(i_res):
        return
    for cnt in range(m):
        color = 'green' if (cnt == i_res[update_cnt]) else 'white'
        myCanvas.create_rectangle(base[0]+cnt*squareL,base[1]+squareL*update_cnt,base[0]+cnt*squareL+squareL,base[1]+squareL*update_cnt+squareL,fill=color)
    for cnt in range(n):
        color = 'red' if (cnt == j_res[update_cnt]) else 'white'
        myCanvas.create_rectangle(base2[0]+cnt*squareL,base2[1]+squareL*update_cnt,base2[0]+cnt*squareL+squareL,base2[1]+squareL*update_cnt+squareL,fill=color)
    myCanvas.pack()
    update_cnt+=1
    root.after(100, update_win)
'''

def update_matrix():
    global update_cnt, root
    if update_cnt == len(i_res):
        return
    for i in range(m):
        for j in range(n):
            color = 'green' if (i == i_res[update_cnt]) and (j == j_res[update_cnt]) else 'white'
            myCanvas.create_rectangle(base[0]+j*squareL, base[1]+i*squareL,base[0]+j*squareL+squareL, base[1]+i*squareL+squareL, fill=color)
    myCanvas.pack()
    string_var.set("Round "+str(update_cnt+1)+": defender chooses " + str(i_res[update_cnt]) + "; attacker chooses " + str(j_res[update_cnt]))
    caption.pack()
    update_cnt+=1
    root.after(500, update_matrix)

def main():
    global i_res, j_res, root, myCanvas, update_cnt, caption, string_var
    rgt = 0.0
    for tr in range(trial):
        update_cnt = 0
        i_res = []
        j_res = []
        R = np.loadtxt("random_instance/R_{}.txt".format(tr))
        C = np.loadtxt("random_instance/C_{}.txt".format(0))
        if tr != 0:
            # re-init tk
            root = tkinter.Tk()
            myCanvas = tkinter.Canvas(root, bg="white", height=imgheight, width=imgwidth)
            string_var = tkinter.StringVar()
            caption = tkinter.Label(root,textvariable = string_var, font=("Arial", 25))
            caption.place(relx = 0.5,
                            rely = 0.9,
                            anchor ='center')
        agent = Follower(utility_matrix=C, behavior_mode=attacker_mode_dict[attacker_mode])  # Instantiate a follower class using the utility matrix and the behavior model
        env = Platform(R, C)
        u_sse = env.compute_SSE()
        if defender_mode == '1':
            x = [np.random.rand() for i in range(m)]
            temp = sum(x)
            x = [i/temp for i in x]
        else:
            x = [1.0/m for _ in range(m)]
        for t in range(T):
            i_t, j_t = env.step(x, agent) # the platform takes a step based on the leader strategy and the follower model
            env.cum_u += R[i_t][j_t]
            i_res.append(i_t)
            j_res.append(j_t)
            print("At trial " + str(tr) + " round " + str(t) + ", the leader plays action " + str(i_t) + " according to " + defender_mode_dict[defender_mode] + str(x) + ", the follower responses by " + str(j_t) + ".")
            print("The leader utility at current round is {}, and the cumulative leader utility until current round is {}.".format(R[i_t][j_t], env.cum_u))
            print("==============================="*5)
            ############################################################
            # the user updates the leader strategy (e.g. random update) #
            ############################################################
            if defender_mode == '1':
                x = [np.random.rand() for i in range(m)]
                temp = sum(x)
                x = [i/temp for i in x]     
            else:
                reward_t = R[:, j_t]
                x = utils.mwu_update(x=x, reward=reward_t, eps=eps)
        print(i_res, j_res)
        update_matrix()
        root.mainloop()
        rgt += (u_sse*T - env.cum_u)/T 
    print("The averaged regret you get over {} trial(s) is {}".format(trial, rgt/trial))

if __name__ == '__main__':
    main()
    #utils.generate_randome_instance(100, 3, 3)