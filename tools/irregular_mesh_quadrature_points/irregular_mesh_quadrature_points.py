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
from matplotlib.gridspec import GridSpec

import meshing.meshbase as mesh_defs
import numerics.basis.basis as basis_defs
import processing.plot as plot

# -- Create two adjacent Q2 elements -- #

# Some points on a parabola
x0, y0 = (-1, 2)
x1, y1 = (-.5, .8)
x2, y2 = (1, 0)
# Create left element
elemL = mesh_defs.Element()
elemL.ID = 0
elemL.node_coords = np.array([
    [-1, 0], [0, 0], [x2, y2],
    [-1, 1], [x1, y1],
    [x0, y0],
    ])
# Create geometric basis to interpolate from reference element to physical space
gorder = 2
gbasis = basis_defs.LagrangeTri(gorder)
# Get basis values at a point (just some point to use as a node)
gbasis_plot = gbasis.get_values(np.array([[.75, .25]]))
# Convert to physical space
x3, y3 = np.matmul(gbasis_plot, elemL.node_coords)[0]
# Create right element
elemR = mesh_defs.Element()
elemR.ID = 1
elemR.node_coords = np.array([
    [x1, y1], [x3, y3], [x2, y2],
    [.5, 1], [1, .5],
    [1, 1],
    ])

# -- Create faces of elements -- #

# Create faces on boundary of elemL
faceL = mesh_defs.InteriorFace()
faceL.refQ1nodes_L = np.array([ [0, 1], [0, 0] ])
faceB = mesh_defs.InteriorFace()
faceB.refQ1nodes_L = np.array([ [0, 0], [1, 0] ])
# Create faces on boundary of elemR
faceR = mesh_defs.InteriorFace()
faceR.refQ1nodes_R = np.array([ [0, 0], [0, 1] ])
faceT = mesh_defs.InteriorFace()
faceT.refQ1nodes_R = np.array([ [0, 1], [1, 0] ])
# Create face between
faceM = mesh_defs.InteriorFace(0, 0, 0, 0)
faceM.refQ1nodes_L = np.array([ [1, 0], [.5, .5] ])
faceM.refQ1nodes_R = np.array([ [1, 0], [0, 0] ])
# Create hanging face for elemL
faceH = mesh_defs.InteriorFace()
faceH.refQ1nodes_L = np.array([ [.5, .5], [0, 1] ])
# Add to elements
elemL.faces = [faceM, faceH, faceL, faceB]
elemR.faces = [faceM, faceR, faceT]

# -- Evaluate midpoint location -- #

# Create geometric basis to interpolate from reference face to reference element
gbasis_face = basis_defs.LagrangeSeg(1)
# Get plotting points on reference face, and evaluate face basis there
x_face_plot = gbasis_face.equidistant_nodes(20)
gbasis_val_face = gbasis_face.get_values(x_face_plot)

xphys = np.empty([7, x_face_plot.shape[0], 2])
x_mid = np.empty([2, 2])
i = 0
# Element node coordinates
elem_coords = elemL.node_coords
# Midpoint of middle face
x_mid_refL = np.mean(faceM.refQ1nodes_L, axis=0, keepdims=True)
# Convert to physical space
x_mid[0] = np.matmul(gbasis.get_values(x_mid_refL), elem_coords)
for face in elemL.faces:
    # Convert reference face to reference element
    x_plot = np.matmul(gbasis_val_face, face.refQ1nodes_L)
    # Get basis values
    gbasis_plot = gbasis.get_values(x_plot)
    # Convert to physical space
    xphys[i] = np.matmul(gbasis_plot, elem_coords)
    i += 1
elem_coords = elemR.node_coords
# Midpoint of middle face
x_mid_refR = np.mean(faceM.refQ1nodes_R, axis=0, keepdims=True)
# Convert to physical space
x_mid[1] = np.matmul(gbasis.get_values(x_mid_refR), elem_coords)
for face in elemR.faces:
    # Convert reference face to reference element
    x_plot = np.matmul(gbasis_val_face, face.refQ1nodes_R)
    # Get basis values
    gbasis_plot = gbasis.get_values(x_plot)
    # Convert to physical space
    xphys[i] = np.matmul(gbasis_plot, elem_coords)
    i += 1

# Difference between physical midpoint locations
print(x_mid[0] - x_mid[1])

# -- Plotting -- #
plot.prepare_plot(linewidth=0.5)
# Plot physical space
fig = plt.figure(constrained_layout=True)
gs = GridSpec(2, 3, figure=fig)
ax0 = fig.add_subplot(gs[:, :2])
# Plot each face and its Q1 nodes
for i in range(xphys.shape[0]):
    ax0.plot(xphys[i, :, 0], xphys[i, :, 1], '-k', lw = 2)
    ax0.plot(xphys[i, [0, -1], 0], xphys[i, [0, -1], 1], 'ob', ms = 10, mfc = 'w')
# Plot midpoints
ax0.plot(x_mid[:, 0], x_mid[:, 1], '.r', ms = 12)
# Label elements
props = dict(boxstyle='round', facecolor='w', alpha=0.5)
ax0.text(-.5, .3, 'L', fontsize = 20, va ='center',
        ha = 'center', bbox=props)
ax0.text(.5, .6, 'R', fontsize = 20, va ='center',
        ha = 'center', bbox=props)
ax0.set_xlabel('$x$')
ax0.set_ylabel('$y$')
# Plot elemL reference space
ax1 = fig.add_subplot(gs[0, 2])
# Plot each face and its Q1 nodes
for face in elemL.faces:
    ax1.plot(face.refQ1nodes_L[:, 0], face.refQ1nodes_L[:, 1], '-k', lw = 2)
    ax1.plot(face.refQ1nodes_L[[0, -1], 0], face.refQ1nodes_L[[0, -1], 1], 'ob',
            ms = 10, mfc = 'w')
# Plot midpoint
ax1.plot(x_mid_refL[0, 0], x_mid_refL[0, 1], '.r', ms = 12)
# Label element
ax1.text(.25, .25, 'L', fontsize = 20, va ='center',
        ha = 'center', bbox=props)
ax1.set_xlabel('$\\xi$')
ax1.set_ylabel('$\\eta$')
# Plot elemR reference space
ax2 = fig.add_subplot(gs[1, 2])
# Plot each face and its Q1 nodes
for face in elemR.faces:
    ax2.plot(face.refQ1nodes_R[:, 0], face.refQ1nodes_R[:, 1], '-k', lw = 2)
    ax2.plot(face.refQ1nodes_R[[0, -1], 0], face.refQ1nodes_R[[0, -1], 1], 'ob',
            ms = 10, mfc = 'w')
# Plot each Q1 node of each face
for i in range(xphys.shape[0]):
    ax0.plot(xphys[i, [0, -1], 0], xphys[i, [0, -1], 1], 'ob', ms = 10, mfc = 'w')
# Plot midpoint
ax2.plot(x_mid_refR[0, 0], x_mid_refR[0, 1], '.r', ms = 12)
# Label element
ax2.text(.25, .25, 'R', fontsize = 20, va ='center',
        ha = 'center', bbox=props)
ax2.set_xlabel('$\\xi$')
ax2.set_ylabel('$\\eta$')
plt.savefig('mesh.svg', bbox_inches='tight')

plt.show()
