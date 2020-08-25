import sys; sys.path.append('../../../../src'); sys.path.append('./src')
import numpy as np
import code
import meshing.common as MeshCommon
import general
import matplotlib as mpl
from matplotlib import pyplot as plt
import processing.plot as Plot
from helper import *
import physics.scalar.scalar as Scalar
import solver.DG as Solver


### Parameters
SolutionOrder = p = 5
basis = General.BasisType["LagrangeEqSeg"]
nL = 51 # number of wavenumbers
alpha = 0. # 0 for upwind flux, 1 for central flux


### Mesh
# Note: one element - same as reference element
mesh = MeshCommon.mesh_1D(Uniform=True, num_elems=1, xmin=0., xmax=1., Periodic=True)
h = mesh.node_coords[-1,0] - mesh.node_coords[-2,0]


### Solver and physics
Params = general.SetSolverParams(SolutionOrder=SolutionOrder,SolutionBasis="LagrangeEqSeg")
physics = Scalar.Scalar(Params["SolutionOrder"], Params["SolutionBasis"], mesh)
physics.set_physical_params(Velocity=1.)
# Initial conditions
physics.IC.Set(Function=physics.FcnExponential)
U = physics.U.Arrays
U[0] = U[0].astype(complex)
solver = Solver.DG(Params,physics,mesh)


### Mass matrix, stiffness matrix, basis
MMinv, SM, PhiLeft, PhiRight, nn = CalculateBasisAndMatrices(mesh, basis, SolutionOrder)


### Allocate
L = np.linspace(0., np.pi, nL)
Omega_r_all = np.zeros([nL, nn])
Omega_i_all = np.zeros_like(Omega_r_all)


### Solve
for i in range(nL):
	Omega_r, Omega_i = GetEigValues(MMinv, SM, PhiLeft, PhiRight, L[i], p, h, alpha, solver)

	Omega_r_all[i,:] = Omega_r[:]
	Omega_i_all[i,:] = Omega_i[:]


### Plot
Plot.prepare_plot()

## Plot dispersion relation
fig0 = plt.figure(0)
for n in range(nn):
	plt.plot(L/np.pi, Omega_r_all[:,n]/nn, 'o', label="%d" % (n))
# Exact
plt.plot(np.array([L[0], L[-1]])/np.pi, np.array([L[0], L[-1]]), ':', label="Exact")
plt.xlabel("$\\Lambda/\\pi$")
plt.ylabel("$\\Omega_r/N_p$")
plt.legend(loc="best")

## Plot dissipation relation
fig1 = plt.figure(1)
for n in range(nn):
	plt.plot(L/np.pi, Omega_i_all[:,n]/nn, 'o', label="%d" % (n))
# Exact
plt.plot(np.array([L[0], L[-1]])/np.pi, np.array([0.,0.]), ':', label="Exact")
plt.xlabel("$\\Lambda/\\pi$")
plt.ylabel("$\\Omega_i/N_p$")
plt.legend(loc="best")

plt.show()

# code.interact(local=locals())


