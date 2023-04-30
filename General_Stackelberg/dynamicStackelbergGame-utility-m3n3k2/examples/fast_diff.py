from dynamic_stackelberg.dyse import dySE, dySE_binary, dySE_fast

R = [[1.0, 1.0], [1.0, 5.0]]
C = [[[4.0, 0.0], [5.0, 4.0]], [[5.0, 1.0], [5.0, 4.0]]]
pi = [0.5, 0.5]
T = 2

M = dySE(R, C, pi, T, False)
for v in M.getVars():
    print(v.varName, v.x)
print("Obj:", M.objVal)

M = dySE_binary(R, C, pi, T, False)
for v in M.getVars():
    print(v.varName, v.x)
print("Obj:", M.objVal)
print(M.getObjective())

M = dySE_fast(R, C, pi, T, False)
for v in M.getVars():
    print(v.varName, v.x)
print("Obj:", M.objVal)
print(M.getObjective())
