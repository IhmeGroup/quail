import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import solver.DG as Solver
import physics.scalar.scalar as Scalar
import meshing.common as MeshCommon
import processing.post as Post
import processing.plot as Plot
import general


### Mesh
Periodic = True
# Uniform mesh
mesh = MeshCommon.mesh_1D(Uniform=True, nElem=32, xmin=-1., xmax=1., Periodic=Periodic)
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
Params = general.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
								 InterpBasis="LegendreSeg",TimeScheme="LSRK4")


### Physics
Velocity = 1.
EqnSet = Scalar.ConstAdvScalar1D(Params["InterpOrder"], Params["InterpBasis"], mesh, StateRank=1)
EqnSet.SetParams(ConstVelocity=Velocity,ConvFlux="LaxFriedrichs")
# Initial conditions
EqnSet.IC.Set(Function=EqnSet.FcnSine, omega = 2*np.pi)
# Exact solution
EqnSet.ExactSoln.Set(Function=EqnSet.FcnSine, omega = 2*np.pi)
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
			BC.Set(Function=EqnSet.FcnSine, BCType=EqnSet.BCType["FullState"], omega = 2*np.pi)
		elif BC.Name is Outflow:
			BC.Set(BCType=EqnSet.BCType["Extrapolation"])
			# BC.Set(Function=EqnSet.FcnSine, BCType=EqnSet.BCType["FullState"], omega = 2*np.pi)
		else:
			raise Exception("BC error")


### Solve
solver = Solver.DG_Solver(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, EqnSet, solver, "Scalar")
# Plot
Plot.PreparePlot()
Plot.PlotSolution(mesh, EqnSet, solver, "Scalar", PlotExact=True, Label="Q_h")
Plot.ShowPlot()


# code.interact(local=locals())
