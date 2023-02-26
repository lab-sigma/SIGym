from dynamic_stackelberg.dyse import dySE_binary

R = [[[0.4, -2.0],
     [-0.2, 1.0]],
     [[0.2, -1.0],
     [-0.5, 1.5]]]
C = [[[-1.5, 1.0],
      [0.8, -1.5]],
     [[-0.2, 1.6],
      [1.2, -0.6]]]
pi = [0.5, 0.5]
T = 4

M = dySE_binary(R, C, pi, T, True)
for v in M.getVars():
    print(v.varName, v.x)
print("Obj:", M.objVal)
