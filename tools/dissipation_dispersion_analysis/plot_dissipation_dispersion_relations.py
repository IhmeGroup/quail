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
#       File : tools/dissipation_dispersion_analysis/
#			plot_dissipation_dispersion_relations.py
#
#       Computes and plots the dissipation and dispersion relations for the
#		DG method. See following reference for details:
#
#		[1] Hesthaven JS, Warburton T. Nodal discontinuous Galerkin methods: 
#		algorithms, analysis, and applications. Springer Science & Business
#		Media; 2007. pp. 88-93.
#      
# ------------------------------------------------------------------------ #
import sys; sys.path.append('../../src');
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from helpers import *

import defaultparams
import meshing.common as mesh_common
import physics.scalar.scalar as scalar
import processing.plot as plot
import solver.DG as DG

import solver.ADERDG as ADERDG


def get_physical_modes(mesh, params, order, nL):
	'''
	This function computes the eigenvalues of the physical modes

    Inputs:
    -------
        mesh: mesh object
        params: dictionary of input parameters
        order: solution order
        nL: number of wavenumbers at which to compute eigenvalues

    Outputs:
    -------- 
        Omega_r_phys: real part of eigenvalues 
        Omega_i_phys: imaginary part of eigenvalues
        nb: number of basis coefficients
        Lplot: normalized wavenumbers for plotting
	'''
	# Set physics
	params["SolutionOrder"] = order 
	physics = scalar.ConstAdvScalar1D()
	physics.set_IC(IC_type="Sine")
	physics.set_conv_num_flux(conv_num_flux_type="LaxFriedrichs")
		# This won't affect results; just to avoid error

	# Set solver
	solver = DG.DG(params, physics, mesh)
	U = solver.state_coeffs
	U = U.astype(complex) # allow for complex numbers

	# Get inverse mass matrix, stiffness matrix, and basis values
	MMinv, SM, basis_valL, basis_valR, nb = get_matrices_and_basis_vals(
			solver, mesh, order)

	# Allocate
	Lplot = np.linspace(0., np.pi, nL - 2)
	dL = Lplot[1] - Lplot[0]
	# Note: Lplot is used for plotting
	# L (below) has additional values for computing derivatives
	L = np.zeros(nL) # Normalized wavenumbers
	L[0] = Lplot[0] - dL; L[-1] = Lplot[-1] + dL
	L[1:-1] = Lplot
	Omega_r_all = np.zeros([nL, nb]) # Real eigenvalue parts
	Omega_i_all = np.zeros_like(Omega_r_all) # Imaginary eigenvalue parts

	# Solve for eigenvalues at each wavenumber
	for i in range(nL):
		Omega_r, Omega_i = get_eig_values(MMinv, SM, basis_valL, basis_valR, 
				L[i], order, h, alpha)

		# Store
		Omega_r_all[i,:] = Omega_r[:]
		Omega_i_all[i,:] = Omega_i[:]


	''' 
	Extract physical mode by comparing second derivatives 
	Note: this is NOT a rigorous way to obtain the physical mode
	'''
	Omega_r_phys = np.zeros(nL - 2)
	Omega_i_phys = np.zeros_like(Omega_r_phys)

	iphys = np.zeros(nL, dtype=int)

	# First three indices - find closest to exact solution
	iphys[0] = (np.abs(Omega_r_all[0, :]/nb - L[0]) + np.abs(
			Omega_i_all[0, :]/nb)).argmin()
	iphys[1] = (np.abs(Omega_r_all[1, :]/nb - L[1]) + np.abs(
			Omega_i_all[1, :]/nb)).argmin()
	iphys[2] = (np.abs(Omega_r_all[2, :]/nb - L[2]) + np.abs(
			Omega_i_all[2, :]/nb)).argmin()

	# Second derivatives of real part
	Omega = Omega_r_all
	d2L = dL**2.
	d2Omega = Omega[2, iphys[2]] - 2.*Omega[1, iphys[1]] + Omega[0, iphys[0]]
	for i in range(2, nL-1):
		deriv2_old = d2Omega/d2L
		# Find closest to deriv2_old
		d2Omegas = Omega[i+1, :] - 2.*Omega[i, iphys[i]] + Omega[i-1, 
				iphys[i-1]]
		iphys[i+1] = np.abs(d2Omegas/d2L - deriv2_old).argmin()
		d2Omega = d2Omegas[iphys[i+1]]

	# Extract
	for i in range(nL - 2):
		Omega_r_phys[i] = Omega_r_all[i+1, iphys[i+1]]
		Omega_i_phys[i] = Omega_i_all[i+1, iphys[i+1]]

	return Omega_r_phys, Omega_i_phys, nb, Lplot


'''
Parameters
'''
# Initialize params dictionary using default parameters (don't change)
params = {**defaultparams.TimeStepping, **defaultparams.Numerics, 
		**defaultparams.Output, **defaultparams.Restart}
params["RestartFile"] = params["File"] # necessary step to avoid error

# The below parameters can be modified
params["SolutionBasis"] = "LagrangeSeg"
params["ElementQuadrature"] = "GaussLegendre"
params["NodeType"] = "Equidistant"

# Uncomment the following block for colocated Gauss-Lobatto scheme
'''
params["ElementQuadrature"] = "GaussLobatto"
params["FaceQuadrature"] = "GaussLobatto"
params["NodeType"] = "GaussLobatto"
params["ColocatedPoints"] = True
'''

# Polynomial orders
orders = range(1, 8)
# Number of wavenumbers
nL = 501
# 0 for upwind flux, 1 for central flux
alpha = 0. 


'''
Pre-processing
'''
norders = len(orders)
# Mesh
mesh = mesh_common.mesh_1D(num_elems=1, xmin=0., xmax=1.)
elem = mesh.elements[0]
h = elem.node_coords[1, 0] - elem.node_coords[0, 0]


'''
Compute
'''
Omega_r_phys_all = np.zeros([nL-2, norders])
Omega_i_phys_all = np.zeros_like(Omega_r_phys_all)
nb_all = np.zeros(norders)
for i in range(norders):
	order = orders[i]
	Omega_r_phys_all[:,i], Omega_i_phys_all[:,i], nb_all[i], Lplot = \
			get_physical_modes(mesh, params, order, nL)


'''
Plot
'''
plot.prepare_plot()

''' Plot dispersion relation '''
plt.figure()
for i in range(norders):
	plt.plot(Lplot/np.pi, Omega_r_phys_all[:,i]/nb_all[i], '--', 
			label="$p = %d$" % (orders[i]))
# Exact relation
plt.plot(np.array([Lplot[0], Lplot[-1]])/np.pi, np.array([Lplot[0], 
		Lplot[-1]]), 'k:', label="Exact")
plt.xlabel("$\\Lambda/\\pi$")
plt.ylabel("$\\Omega_r/N_p$")
plt.legend(loc="best")
plot.save_figure(file_name='Dispersion', file_type='pdf', crop_level=2)

''' Plot dissipation relation '''
plt.figure()
for i in range(norders):
	plt.plot(Lplot[:-1]/np.pi, Omega_i_phys_all[:-1,i]/nb_all[i], '--', 
			label="$p = %d$" % (orders[i]))
# Exact relation
plt.plot(np.array([Lplot[0], Lplot[-1]])/np.pi, np.array([0., 0.]), 
		'k:', label="Exact")
plt.xlabel("$\\Lambda/\\pi$")
plt.ylabel("$\\Omega_i/N_p$")
plt.legend(loc="best")
breakpoint()

plot.save_figure(file_name='Dissipation', file_type='pdf', crop_level=2)

plt.show()