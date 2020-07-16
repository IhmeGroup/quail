import sys; sys.path.append('../src'); sys.path.append('./src')
import code
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import general
import processing.plot as Plot
import numerics.basis.basis as Basis


### Grid
n = 101 # number of points in each direction
x = y = np.linspace(-1., 1., n)
X,Y = np.meshgrid(x,y)
xp = np.array([np.reshape(X,-1),np.reshape(Y,-1)]).transpose()
ntot = n**2 # total number of points

### Parameters
p = 3 # Order
b = 11 # which basis fcn to plot

basis = Basis.LagrangeQuad(p)
### Evaluate basis functions
basis.get_basis_val_grads(xp, get_val=True)
# Reshape
Z = basis.basis_val[:,b-1]
Z.shape = n,-1


### Plot
Plot.PreparePlot(linewidth=0.5)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap='plasma', rstride=1, cstride=1) # cmap=autumn_r
ax.set_xlabel('$\\xi$')
ax.set_ylabel('$\\eta$')
ax.set_zlabel('$\\phi$')
plt.show()
