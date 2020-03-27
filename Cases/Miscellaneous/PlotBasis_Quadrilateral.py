import sys; sys.path.append('../../src'); sys.path.append('./src')
import numpy as np
import code
import Plot
import General
from Basis import BasisData
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


### Grid
n = 101 # number of points in each direction
x = y = np.linspace(-1., 1., n)
X,Y = np.meshgrid(x,y)
xp = np.array([np.reshape(X,-1),np.reshape(Y,-1)]).transpose()
ntot = n**2 # total number of points


### Parameters
p = 3 # Order
b = 11 # which basis fcn to plot


### Evaluate basis functions
PhiData = BasisData(General.BasisType.LagrangeQuad, p, nq = ntot)
PhiData.EvalBasis(xp, Get_Phi=True)
# Reshape
Z = PhiData.Phi[:,b-1]
Z.shape = n,-1


### Plot
Plot.PreparePlot(linewidth=0.5)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap='plasma', rstride=1, cstride=1) # cmap=autumn_r
ax.set_xlabel('$\\xi$')
ax.set_ylabel('$\\eta$')
ax.set_zlabel('$\\phi$')
# ax.set_zlim([0, 1])
plt.show()

# code.interact(local=locals())



