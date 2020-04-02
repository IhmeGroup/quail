import sys; sys.path.append('../../src'); sys.path.append('./src')
import numpy as np
import code
import Plot
import General
from Basis import BasisData, equidistant_nodes
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


### Parameters
# basis type
# basis = General.BasisType.LagrangeQuad 
basis = General.BasisType.LagrangeTri
p = 1 # Order
b = 3 # which basis fcn to plot


### Grid
p_plot = 50 # plot at Lagrange nodes corresponding to p_plot
xp, np = equidistant_nodes(basis, p_plot)


### Evaluate basis functions
PhiData = BasisData(basis, p, nq = np)
PhiData.EvalBasis(xp, Get_Phi=True)
# Reshape
phi = PhiData.Phi[:,b-1]
# Z.shape = n,-1


### Plot
Plot.PreparePlot(linewidth=0.5)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(xp[:,0], xp[:,1], phi, cmap='plasma')
ax.set_xlabel('$\\xi$')
ax.set_ylabel('$\\eta$')
ax.set_zlabel('$\\phi$')
# ax.set_zlim([0, 1])
plt.show()

# code.interact(local=locals())



