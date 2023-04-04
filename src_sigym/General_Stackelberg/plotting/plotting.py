import matplotlib.pyplot as plt
import numpy as np


fig, (ax3) = plt.subplots(1,1)
#fig, (ax1, ax2, ax3) = plt.subplots(1,3)
fig.suptitle('The allocation, payment, and utility function of different kappa values')

x = [0, 1, 2, 3, 4, 5]
bss = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
dyss = [0.0, 0.5, 1.4583333333333300, 2.25, 3.25, 4.0]
mdss = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
mrss = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

#ax3.plot(x, r1, label = "kappa_0.0")
#ax3.plot(x, r3, label = "kappa_0.05")
#ax3.plot(x, r2, label = "kappa_0.2")
ax3.plot(x, bss, label = "test1")
ax3.plot(x, dyss, label = "test2")
ax3.plot(x, mdss, label = "test1")
ax3.plot(x, mrss, label = "test2")
# ax3.plot(x, y7, label = "kappa_0.05_lip")
# ax3.plot(x, y2, label = "kappa_0.05")
# #ax3.plot(x, y3, label = "kappa_0.05")
# #ax3.plot(x, y4, label = "kappa_0.05")
# ax3.plot(x, y5, label = "kappa_0.05")
# ax3.plot(x, y6, label = "kappa_0.05_lip")
ax3.legend(loc="upper right")

# ax2.plot(x8, y8, label = "p1: [1.0, 0.0]")
# ax2.plot(x7, y7, label = "p1: [0.75, 0.25]")
# ax2.plot(x1, y1, label = "p1: [0.5, 0.5]")
# ax2.plot(x6, y6, label = "p1: [0.25, 0.75]")
# ax2.plot(x9, y9, label = "p1: [0.0, 1.0]")
# ax2.legend(loc='upper right')

# ax3.plot(xa, ya, label = "v1: [low = 0.0, high = 0.0]")
# ax3.plot(x1, y1, label = "v1: [low = 0.0, high = 1.5]")
# ax3.plot(xb, yb, label = "v1: [low = 0.0, high = 2.5]")
# ax3.plot(xe, ye, label = "v1: [low = 1.0, high = 2.0]")
# ax3.plot(xd, yd, label = "v1: [low = 1.5, high = 3.0]")
# ax3.plot(xc, yc, label = "v1: [low = 2.0, high = 3.0]")
# ax3.legend(loc='upper right')
#plt.plot(y, x, label = "line 2")
#plt.plot(x, np.sin(x), label = "curve 1")
#plt.plot(x, np.cos(x), label = "curve 2")
#plt.legend()
plt.show()