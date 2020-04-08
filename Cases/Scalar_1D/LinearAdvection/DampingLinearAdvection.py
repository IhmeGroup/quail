import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import Solver
import Scalar
import MeshCommon
import Post
import Plot
import General


### Mesh
Periodic = False
# Uniform mesh
mesh = MeshCommon.mesh_1D(Uniform=True, nElem=16, xmin=-1., xmax=1., Periodic=Periodic)
# Non-uniform mesh
# nElem = 25
# Coords = np.cos(np.linspace(np.pi,0.,nElem+1))
# Coords = MeshCommon.refine_uniform_1D(Coords)
# # Coords = MeshCommon.refine_uniform_1D(Coords)
# mesh = MeshCommon.mesh_1D(Coords=Coords, Periodic=Periodic)


### Solver parameters
EndTime = 0.5
nTimeStep = np.amax([1,int(EndTime/((mesh.Coords[1,0] - mesh.Coords[0,0])*0.1))])
InterpOrder = 2
Params = General.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
								 InterpBasis="LagrangeSeg",TimeScheme="RK4",InterpolateFlux=True)
nu = -3.

### Physics
Velocity = 1.0 
EqnSet = Scalar.Scalar(Params["InterpOrder"], Params["InterpBasis"], mesh, StateRank=1)
EqnSet.SetParams(ConstVelocity=Velocity)
EqnSet.SetParams(ConvFlux="LaxFriedrichs")
EqnSet.SetSource(Function=EqnSet.FcnSimpleSource, nu = nu)


Uinflow=[1.]
# Initial conditions
EqnSet.IC.Set(Function=EqnSet.FcnDampingSine, omega = 2.*np.pi , nu = nu)
# Exact solution
EqnSet.ExactSoln.Set(Function=EqnSet.FcnDampingSine, omega = 2.*np.pi , nu = nu)
# Boundary conditions
if Velocity >= 0.:
	Inflow = "Left"; Outflow = "Right"
else:
	Inflow = "Right"; Outflow = "Left"
if not Periodic:
	for ibfgrp in range(mesh.nBFaceGroup):
		BC = EqnSet.BCs[ibfgrp]
		## Left
		if BC.Name is Inflow:
			BC.Set(Function=EqnSet.FcnDampingSine, BCType=EqnSet.BCType["FullState"], omega = 2.*np.pi, nu=nu)
		elif BC.Name is Outflow:
			BC.Set(BCType=EqnSet.BCType["Extrapolation"])
			#BC.Set(Function=EqnSet.FcnDampingSine, BCType=EqnSet.BCType["FullState"], omega = 2*np.pi, nu=-2.0)
		else:
			raise Exception("BC error")


### Solve
solver = Solver.DG_Solver(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, EqnSet, solver.Time, "Scalar")
# Plot
Plot.PreparePlot()
Plot.PlotSolution(mesh, EqnSet, solver.Time, "Scalar", PlotExact=True, PlotIC=True, Label="u")
Plot.ShowPlot()


# code.interact(local=locals())
