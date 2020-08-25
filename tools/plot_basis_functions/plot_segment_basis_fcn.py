import sys; sys.path.append('../../src')
import code
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import general

import numerics.basis.basis as basis_defs
import numerics.basis.tools as basis_tools

import processing.plot as plot


'''
Parameters, initialization
'''
p = 4 # order
plot_all = True
b = 0 # the (b+1)th basis function will be plotted if plot_all is False
node_type = "GaussLobatto"
basis = basis_defs.LagrangeSeg(p)
# basis = basis_defs.LegendreSeg(p)
basis.get_1d_nodes = basis_tools.set_1D_node_calc(node_type)


'''
Get sample points
'''
p_plot = 100 # (p_plot+1) points
xp = basis.equidistant_nodes(p_plot)


'''
Evaluate
'''
basis.get_basis_val_grads(xp, get_val=True)


'''
Plot
'''
plot.prepare_plot(linewidth=1.)
fig = plt.figure()
for i in range(p+1):
	if plot_all or (not plot_all and i == b):
		phi = basis.basis_val[:, i]
		plt.plot(xp, phi, label="$\\phi_{%d}$" % (i+1))
plt.xlabel('$\\xi$')
plt.ylabel('$\\phi$')
plt.legend(loc="best")
plt.xticks((-1., -0.5, 0., 0.5, 1.))
plt.show()