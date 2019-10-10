import sys; sys.path.append('../../../../src'); sys.path.append('./src')
import numpy as np
import code
import MeshCommon
import General
import matplotlib as mpl
from matplotlib import pyplot as plt
import Plot
from helper import *


### Parameters
InterpOrder = p = 5
basis = General.BasisType["SegLagrange"]
nL = 51 # number of wavenumbers


### Mesh
# Note: dummy mesh
mesh = MeshCommon.Mesh1D(Uniform=True, nElem=2, xmin=-1., xmax=1., Periodic=True)
h = mesh.Coords[1,0] - mesh.Coords[0,0]


### Mass matrix, stiffness matrix, basis
MMinv, SM, PhiLeft, PhiRight, nn = CalculateBasisAndMatrices(mesh, basis, InterpOrder)


### Allocate
L = np.linspace(0., np.pi, nL)
Omega_r_all = np.zeros([nL, nn])
Omega_i_all = np.zeros_like(Omega_r_all)


### Solve
for i in range(nL):
	Omega_r, Omega_i = GetEigValues(MMinv, SM, PhiLeft, PhiRight, L[i], p, h)

	Omega_r_all[i,:] = Omega_r[:]
	Omega_i_all[i,:] = Omega_i[:]


### Plot
Plot.PreparePlot()

## Plot dispersion relation
fig0 = plt.figure(0)
for n in range(nn):
	plt.plot(L/np.pi, Omega_r_all[:,n]/nn, 'o', label="Mode %d" % (n))
# Exact
plt.plot(np.array([L[0], L[-1]])/np.pi, np.array([L[0], L[-1]]), ':', label="Exact")
plt.xlabel("$L/\\pi$")
plt.ylabel("$\\Omega_r/N_p$")
plt.legend(loc="best")
plt.show()

## Plot dissipation relation
fig1 = plt.figure(1)
for n in range(nn):
	plt.plot(L/np.pi, Omega_i_all[:,n]/nn, 'o', label="Mode %d" % (n))
# Exact
plt.plot(np.array([L[0], L[-1]])/np.pi, np.array([0.,0.]), ':', label="Exact")
plt.xlabel("$L/\\pi$")
plt.ylabel("$\\Omega_i/N_p$")
plt.legend(loc="best")
plt.show()

# code.interact(local=locals())


