import sys; sys.path.append('../src'); sys.path.append('./src')
import code
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import processing.plot as Plot
import general
import numerics.basis.basis as Basis

### Parameters
# basis type
# basis = General.BasisType.LagrangeEqQuad 

p = 1 # Order
b = 3 # which basis fcn to plot
basis = Basis.LagrangeEqTri(p)


### Grid
p_plot = 50 # plot at Lagrange nodes corresponding to p_plot
xp, npoint = basis.equidistant_nodes(p_plot)

### Evaluate basis functions
basis.eval_basis(xp, Get_Phi=True)
# Reshape
phi = basis.basis_val[:,b-1]

### Plot
Plot.PreparePlot(linewidth=0.5)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(xp[:,0], xp[:,1], phi, cmap='plasma')
ax.set_xlabel('$\\xi$')
ax.set_ylabel('$\\eta$')
ax.set_zlabel('$\\phi$')
plt.show()


