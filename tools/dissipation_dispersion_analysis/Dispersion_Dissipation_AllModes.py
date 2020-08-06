import sys; sys.path.append('../../src'); # sys.path.append('./src')
import code
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from helper import *

import defaultparams
import general
import meshing.common as mesh_common
import physics.scalar.scalar as scalar
import processing.plot as plot
import solver.DG as DG
import numerics.basis.basis as basis_defs


def get_physical_modes(mesh, params, order, nL):
	params["InterpOrder"] = order 
	# physics = Scalar.Scalar(Params["InterpOrder"], Params["InterpBasis"], mesh)
	physics = scalar.ConstAdvScalar1D(order, params["InterpBasis"], mesh)
	# physics.set_physical_params(Velocity=1.)

	# Initial conditions
	# physics.IC.Set(Function=physics.FcnExponential)
	physics.set_IC(IC_type="Sine")
	physics.set_conv_num_flux(conv_num_flux_type="LaxFriedrichs")
	U = physics.U
	U = U.astype(complex)

	# Set solver
	solver = DG.DG(params, physics, mesh)


	### Mass matrix, stiffness matrix, basis
	MMinv, SM, PhiLeft, PhiRight, nn = CalculateBasisAndMatrices(solver, mesh, order)

	### Allocate
	Lplot = np.linspace(0., np.pi, nL - 2)
	dL = Lplot[1] - Lplot[0]
	L = np.zeros(nL)
	L[0] = Lplot[0] - dL; L[-1] = Lplot[-1] + dL
	L[1:-1] = Lplot
	Omega_r_all = np.zeros([nL, nn])
	Omega_i_all = np.zeros_like(Omega_r_all)


	### Solve
	for i in range(nL):
		Omega_r, Omega_i = GetEigValues(MMinv, SM, PhiLeft, PhiRight, L[i], order, h, alpha, solver)

		Omega_r_all[i,:] = Omega_r[:]
		Omega_i_all[i,:] = Omega_i[:]


	### Extract physical mode
	Omega_r_phys = np.zeros(nL - 2)
	Omega_i_phys = np.zeros_like(Omega_r_phys)

	iphys = np.zeros(nL, dtype=int)

	# First three indices - find closest to exact solution
	Omega = Omega_r_all
	iphys[0] = np.abs(Omega[0, :]/nn - L[0]).argmin()
	iphys[1] = np.abs(Omega[1, :]/nn - L[1]).argmin()
	iphys[2] = np.abs(Omega[2, :]/nn - L[2]).argmin()
	# iphys[0] = np.abs(Omega_r_all[1, :]/nn).argmin()
	# Omega_r_phys[0] = Omega_r_all[1, iphys[0]]
	# Omega_i_phys[0] = Omega_i_all[1, iphys[0]]

	# Second derivative
	d2L = dL**2.
	d2Omega = Omega[2, iphys[2]] - 2.*Omega[1, iphys[1]] \
			+ Omega[0, iphys[0]]
	for i in range(2, nL-1):
		# if i == nL-3:
		# 	Omega = Omega_i_all
		# 	d2Omega = Omega[i, iphys[i]] - 2.*Omega[i-1, iphys[i-1]] \
		# 		+ Omega[i-2, iphys[i-2]]
		deriv2_old = d2Omega/d2L
		# Find closest to deriv2_old
		d2Omegas = Omega[i+1, :] - 2.*Omega[i, iphys[i]] \
				+ Omega[i-1, iphys[i-1]]
		iphys[i+1] = np.abs(d2Omegas/d2L - deriv2_old).argmin()
		d2Omega = d2Omegas[iphys[i+1]]

	for i in range(nL - 2):
		Omega_r_phys[i] = Omega_r_all[i+1, iphys[i+1]]
		Omega_i_phys[i] = Omega_i_all[i+1, iphys[i+1]]
	# Omega_r_phys[-1] = 0.

	return Omega_r_phys, Omega_i_phys, nn, Lplot


### Parameters
orders = range(1, 8)
norders = len(orders)
# basis = Basis.LagrangeEqSeg(order)
nL = 501 # number of wavenumbers
alpha = 0. # 0 for upwind flux, 1 for central flux

### Mesh
# Note: one element - same as reference element
mesh = mesh_common.mesh_1D(Uniform=True, nElem=1, xmin=0., xmax=1., Periodic=True)
h = mesh.node_coords[1,0] - mesh.node_coords[0,0]


### Solver and physics
# Params = general.SetSolverParams(InterpOrder=InterpOrder,InterpBasis="LagrangeEqSeg")
params = {**defaultparams.TimeStepping, **defaultparams.Numerics, **defaultparams.Output, **defaultparams.Restart}
# params["InterpOrder"] = order 
params["RestartFile"] = params["File"]
params["InterpBasis"] = "LagrangeSeg"
# params["ElementQuadrature"] = "GaussLobatto"
# params["FaceQuadrature"] = "GaussLobatto"
# params["NodeType"] = "GaussLobatto"
params["ElementQuadrature"] = "GaussLegendre"
params["FaceQuadrature"] = "GaussLegendre"
params["NodeType"] = "Equidistant"
params["NodesEqualQuadpts"] = False
# # physics = Scalar.Scalar(Params["InterpOrder"], Params["InterpBasis"], mesh)
# physics = scalar.ConstAdvScalar1D(order, params["InterpBasis"], mesh)
# # physics.set_physical_params(Velocity=1.)

# # Initial conditions
# # physics.IC.Set(Function=physics.FcnExponential)
# physics.set_IC(IC_type="Sine")
# physics.set_conv_num_flux(conv_num_flux_type="LaxFriedrichs")
# U = physics.U
# U = U.astype(complex)

# # Set solver
# solver = DG.DG(params, physics, mesh)


# ### Mass matrix, stiffness matrix, basis
# MMinv, SM, PhiLeft, PhiRight, nn = CalculateBasisAndMatrices(solver, mesh, order)


# ### Allocate
# Lplot = np.linspace(0., np.pi, nL - 2)
# dL = Lplot[1] - Lplot[0]
# L = np.zeros(nL)
# L[0] = Lplot[0] - dL; L[-1] = Lplot[-1] + dL
# L[1:-1] = Lplot
# Omega_r_all = np.zeros([nL, nn])
# Omega_i_all = np.zeros_like(Omega_r_all)


# ### Solve
# for i in range(nL):
# 	Omega_r, Omega_i = GetEigValues(MMinv, SM, PhiLeft, PhiRight, L[i], order, h, alpha, solver)

# 	Omega_r_all[i,:] = Omega_r[:]
# 	Omega_i_all[i,:] = Omega_i[:]


# ### Extract physical mode
# Omega_r_phys = np.zeros(nL - 2)
# Omega_i_phys = np.zeros_like(Omega_r_phys)

# iphys = np.zeros(nL, dtype=int)

# # First three indices - find closest to exact solution
# iphys[0] = np.abs(Omega_r_all[0, :]/nn - L[0]).argmin()
# iphys[1] = np.abs(Omega_r_all[1, :]/nn - L[1]).argmin()
# iphys[2] = np.abs(Omega_r_all[2, :]/nn - L[2]).argmin()
# # iphys[0] = np.abs(Omega_r_all[1, :]/nn).argmin()
# # Omega_r_phys[0] = Omega_r_all[1, iphys[0]]
# # Omega_i_phys[0] = Omega_i_all[1, iphys[0]]

# # Second derivative
# d2L = dL**2.
# d2Omega = Omega_r_all[2, iphys[2]] - 2.*Omega_r_all[1, iphys[1]] \
# 		+ Omega_r_all[0, iphys[0]]
# for i in range(2, nL-1):
# 	deriv2_old = d2Omega/d2L
# 	# Find closest to deriv2_old
# 	d2Omegas = Omega_r_all[i+1, :] - 2.*Omega_r_all[i, iphys[i]] \
# 			+ Omega_r_all[i-1, iphys[i-1]]
# 	iphys[i+1] = np.abs(d2Omegas/d2L - deriv2_old).argmin()
# 	d2Omega = d2Omegas[iphys[i+1]]

# for i in range(nL - 2):
# 	Omega_r_phys[i] = Omega_r_all[i+1, iphys[i+1]]
# 	Omega_i_phys[i] = Omega_i_all[i+1, iphys[i+1]]

Omega_r_phys_all = np.zeros([nL-2, norders])
Omega_i_phys_all = np.zeros_like(Omega_r_phys_all)
nb_all = np.zeros(norders)
for i in range(norders):
	order = orders[i]
	Omega_r_phys_all[:,i], Omega_i_phys_all[:,i], nb_all[i], Lplot = get_physical_modes(mesh, params, order, nL)


## Plot
plot.PreparePlot()

## Plot dispersion relation
plt.figure()
# for n in range(nn):
for i in range(norders):
	plt.plot(Lplot/np.pi, Omega_r_phys_all[:,i]/nb_all[i], '--', label="$p = %d$" % (orders[i]))
# Exact
plt.plot(np.array([Lplot[0], Lplot[-1]])/np.pi, np.array([Lplot[0], Lplot[-1]]), 'k:', label="Exact")
plt.xlabel("$\\Lambda/\\pi$")
plt.ylabel("$\\Omega_r/N_p$")
plt.legend(loc="best")
plot.SaveFigure(FileName='Dispersion', FileType='pdf', CropLevel=2)

# ## Plot dissipation relation
plt.figure()
# for n in range(nn):
for i in range(norders):
	plt.plot(Lplot[:-1]/np.pi, Omega_i_phys_all[:-1,i]/nb_all[i], '--', label="$p = %d$" % (orders[i]))
# Exact
plt.plot(np.array([Lplot[0], Lplot[-1]])/np.pi, np.array([0., 0.]), 'k:', label="Exact")
plt.xlabel("$\\Lambda/\\pi$")
plt.ylabel("$\\Omega_i/N_p$")
plt.legend(loc="best")
plot.SaveFigure(FileName='Dissipation', FileType='pdf', CropLevel=2)
# fig1 = plt.figure(1)
# for n in range(nn):
# 	plt.plot(L/np.pi, Omega_i_all[:,n]/nn, 'o', label="%d" % (n))
# # Exact
# plt.plot(np.array([L[0], L[-1]])/np.pi, np.array([0.,0.]), ':', label="Exact")
# plt.xlabel("$\\Lambda/\\pi$")
# plt.ylabel("$\\Omega_i/N_p$")
# plt.legend(loc="best")

plt.show()

# code.interact(local=locals())


