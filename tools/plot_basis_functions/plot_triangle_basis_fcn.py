import sys; sys.path.append('../../src')
import code
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import general
import numerics.basis.basis as basis_defs
import processing.plot as plot


'''
Parameters, initialization
'''
p = 1 # order
b = 2 # the (b+1)th basis function will be plotted
basis = basis_defs.LagrangeTri(p)


'''
Get sample points
'''
p_plot = 50 # (p_plot+1) points in each direction
xp = basis.equidistant_nodes(p_plot)


'''
Evaluate
'''
basis.get_basis_val_grads(xp, get_val=True)
phi = basis.basis_val[:, b]


'''
Plot
'''
plot.prepare_plot(linewidth=0.5)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(xp[:, 0], xp[:, 1], phi, cmap='plasma')
ax.set_xlabel('$\\xi$')
ax.set_ylabel('$\\eta$')
ax.set_zlabel('$\\phi_{%d}$' % (b+1))
plt.show()