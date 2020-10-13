# ------------------------------------------------------------------------ #
#
#       File : tools/plot_basis_functions/plot_quadrilateral_basis_fcn.py
#
#       Plots basis functions for the reference quadrilateral.
#      
# ------------------------------------------------------------------------ #
import sys; sys.path.append('../../src')
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import general

import numerics.basis.basis as basis_defs
import numerics.basis.tools as basis_tools

import processing.plot as plot


'''
Parameters
'''
p = 3 # polynomial order
b = 10 # the (b+1)th basis function will be plotted

# Node type (matters for nodal basis only)
node_type = "Equidistant"
# node_type = "GaussLobatto"

# Basis type
basis = basis_defs.LagrangeQuad(p) # Lagrange basis
# basis = basis_defs.LegendreQuad(p) # Legendre basis


'''
Pre-processing
'''
# Solution nodes
basis.get_1d_nodes = basis_tools.set_1D_node_calc(node_type)

# Sample points
n = 101 # number of points in each direction
x = y = np.linspace(-1., 1., n)
X, Y = np.meshgrid(x, y)
xp = np.array([np.reshape(X, -1), np.reshape(Y, -1)]).transpose()
ntot = n**2 # total number of points


'''
Evaluate
'''
basis.get_basis_val_grads(xp, get_val=True)
# Reshape
Z = basis.basis_val[:, b].reshape(n, -1)


'''
Plot
'''
plot.prepare_plot(linewidth=0.5)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap='plasma', rstride=1, cstride=1) # cmap=autumn_r
ax.set_xlabel('$\\xi$')
ax.set_ylabel('$\\eta$')
ax.set_zlabel('$\\phi_{%d}$' % (b+1))
plt.xticks((-1., -0.5, 0., 0.5, 1.))
plt.yticks((-1., -0.5, 0., 0.5, 1.))
plt.show()