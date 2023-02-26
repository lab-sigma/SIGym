from dynamic_stackelberg.dyse import dySE_binary

R = [[1.0, 0.6],
     [0.4, 1.0]]
C = [[[1.0, 0.5],
      [0.5, 0.1]],
     [[0.1, 0.5],
      [0.2, 1.0]]]
pi = [0.5, 0.5]
T = 2

M = dySE_binary(R, C, pi, T, False)
for v in M.getVars():
    print(v.varName, v.x)
print("Obj:", M.objVal)
