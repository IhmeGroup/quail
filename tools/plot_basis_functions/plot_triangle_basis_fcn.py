# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#		<https://github.com/IhmeGroup/quail>
#       
#		Copyright (C) 2020-2021
#
#       This program is distributed under the terms of the GNU
#		General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.  
#		If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#
#       File : tools/plot_basis_functions/plot_triangle_basis_fcn.py
#
#       Plots basis functions for the reference triangle.
#      
# ------------------------------------------------------------------------ #
import sys; sys.path.append('../../src')
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import numerics.basis.basis as basis_defs
import processing.plot as plot


'''
Parameters
'''
p = 1 # polynomial order
b = 2 # the (b+1)th basis function will be plotted

# Basis type
basis = basis_defs.LagrangeTri(p)
# basis = basis_defs.HierarchicH1Tri(p)


'''
Pre-processing
'''
# Sample points
p_plot = 50 # (p_plot+1) points in each direction
xp = basis.equidistant_nodes(p_plot)


'''
Evaluate
'''
basis.get_basis_val_grads(xp, get_val=True)
vals = basis.basis_val[:, b]


'''
Plot
'''
plot.prepare_plot(linewidth=0.5)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(xp[:, 0], xp[:, 1], vals, cmap='plasma')
ax.set_xlabel('$\\xi$')
ax.set_ylabel('$\\eta$')
ax.set_zlabel('$\\phi_{%d}$' % (b+1))
plt.show()