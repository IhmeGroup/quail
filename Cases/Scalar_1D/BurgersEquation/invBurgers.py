import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import Solver
import Scalar
import MeshCommon
import Post
import Plot
import General
import Limiter
### Mesh
Periodic = False
# Uniform mesh
mesh = MeshCommon.mesh_1D(Uniform=True, nElem=2, xmin=-1., xmax=1., Periodic=Periodic)
# Non-uniform mesh
# nElem = 25
# Coords = np.cos(np.linspace(np.pi,0.,nElem+1))
# Coords = MeshCommon.refine_uniform_1D(Coords)
# # Coords = MeshCommon.refine_uniform_1D(Coords)
# mesh = MeshCommon.mesh_1D(Coords=Coords, Periodic=Periodic)


### Solver parameters
#dt = 0.001
#mu = 1.
EndTime = 0.1
nTimeStep = np.amax([1,int(EndTime/((mesh.Coords[1,0] - mesh.Coords[0,0])*0.01))])
#nTimeStep = int(EndTime/dt)
InterpOrder = 3
Params = General.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
								 InterpBasis="LagrangeEqSeg",TimeScheme="ADER")
### Physics
ConstVelocity = 1.
EqnSet = Scalar.Burgers(Params["InterpOrder"], Params["InterpBasis"], mesh, StateRank=1)
# EqnSet.SetParams(AdvectionOperator="Burgers")
EqnSet.SetParams(ConstVelocity=ConstVelocity, ConvFlux="LaxFriedrichs")

# Initial conditions
EqnSet.IC.Set(Function=EqnSet.FcnLinearBurgers)
# Exact solution
EqnSet.ExactSoln.Set(Function=EqnSet.FcnLinearBurgers)
# Boundary conditions
if ConstVelocity >= 0.:
	Inflow = "Left"; Outflow = "Right"
else:
	Inflow = "Right"; Outflow = "Left"
if not Periodic:
	for ibfgrp in range(mesh.nBFaceGroup):
		BC = EqnSet.BCs[ibfgrp]
		## Left
		if BC.Name is Inflow:
			BC.Set(Function=EqnSet.FcnLinearBurgers, BCType=EqnSet.BCType["FullState"])
		elif BC.Name is Outflow:
			BC.Set(BCType=EqnSet.BCType["Extrapolation"])
			# BC.Set(Function=EqnSet.FcnSine, BCType=EqnSet.BCType["FullState"], omega = 2*np.pi)
		else:
			raise Exception("BC error")


### Solve
solver = Solver.ADERDG_Solver(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, EqnSet, solver, "Scalar")
# Plot
Plot.PreparePlot()
Plot.PlotSolution(mesh, EqnSet, solver, "Scalar", PlotExact = True, PlotIC = True, Label="u")
Plot.ShowPlot()


# code.interact(local=locals())
