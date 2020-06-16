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
InterpOrder = p = 5
basis = General.BasisType["LagrangeEqSeg"]
nL = 51 # number of wavenumbers
alpha = 0. # 0 for upwind flux, 1 for central flux


### Mesh
# Note: one element - same as reference element
mesh = MeshCommon.mesh_1D(Uniform=True, nElem=1, xmin=0., xmax=1., Periodic=True)
h = mesh.Coords[1,0] - mesh.Coords[0,0]


### Solver and physics
Params = general.SetSolverParams(InterpOrder=InterpOrder,InterpBasis="LagrangeEqSeg")
EqnSet = Scalar.Scalar(Params["InterpOrder"], Params["InterpBasis"], mesh)
EqnSet.SetParams(Velocity=1.)
# Initial conditions
EqnSet.IC.Set(Function=EqnSet.FcnExponential)
U = EqnSet.U.Arrays
U[0] = U[0].astype(complex)
solver = Solver.DG(Params,EqnSet,mesh)


### Mass matrix, stiffness matrix, basis
MMinv, SM, PhiLeft, PhiRight, nn = CalculateBasisAndMatrices(mesh, basis, InterpOrder)


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
Plot.PreparePlot()

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


