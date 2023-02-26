from dynamic_stackelberg.dyse import dySE, dySE_binary, dySE_fast

R = [[0.5, 0.6],
     [0.1, 1.0]]
C = [[[0.2, 0.5],
      [0.8, 0.5]],
     [[1.0, 0.5],
      [0.25, 0.5]]]
pi = [0.5, 0.5]
T = 4

M = dySE(R, C, pi, T, False)
for v in M.getVars():
    print(v.varName, v.x)
print("Obj:", M.objVal)

M = dySE_binary(R, C, pi, T, False)
for v in M.getVars():
    print(v.varName, v.x)
print("Obj:", M.objVal)

M = dySE_fast(R, C, pi, T, False)
for v in M.getVars():
    print(v.varName, v.x)
print("Obj:", M.objVal)
