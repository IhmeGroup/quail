import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import solver.ADERDG as Solver
import physics.scalar.scalar as Scalar
import meshing.common as MeshCommon
import processing.post as Post
import processing.plot as Plot
import general


### Mesh
Periodic = False 
# Uniform mesh
mesh = MeshCommon.mesh_1D(Uniform=True, nElem=32, xmin=-1., xmax=1., Periodic=Periodic)
# Non-uniform mesh
# nElem = 25
# Coords = np.cos(np.linspace(np.pi,0.,nElem+1))
# Coords = MeshCommon.refine_uniform_1D(Coords)
# # Coords = MeshCommon.refine_uniform_1D(Coords)
# mesh = MeshCommon.mesh_1D(Coords=Coords, Periodic=Periodic)


### Solver parameters
EndTime = 0.1
nTimeStep = np.amax([1,int(EndTime/((mesh.Coords[1,0] - mesh.Coords[0,0])*0.1))])
InterpOrder = 2
Params = general.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
								 InterpBasis="LagrangeEqSeg",TimeScheme="ADER")


### Physics
Velocity = 1.0
EqnSet = Scalar.ConstAdvScalar1D(Params["InterpOrder"], Params["InterpBasis"], mesh)
EqnSet.SetParams(ConstVelocity=Velocity)
#EqnSet.SetParams(AdvectionOperator="Burgers")
EqnSet.SetParams(ConvFlux="LaxFriedrichs")
# Initial conditions
# EqnSet.IC.Set(Function=EqnSet.FcnSine, omega = 2*np.pi)
EqnSet.set_IC(IC_type="Sine", omega = 2*np.pi)
# Exact solution
# EqnSet.ExactSoln.Set(Function=EqnSet.FcnSine, omega = 2*np.pi)
EqnSet.set_exact(exact_type="Sine", omega = 2*np.pi)
# Boundary conditions
if Velocity >= 0.:
	Inflow = "Left"; Outflow = "Right"
else:
	Inflow = "Right"; Outflow = "Left"
	set_BC(self, BC_name, **kwargs)
if not Periodic:
	for ibfgrp in range(mesh.nBFaceGroup):
		BFG = mesh.BFaceGroups[ibfgrp]
		if BFG.Name is Inflow:
			EqnSet.set_BC(BC_type="FullState", fcn_type="Sine", omega = 2*np.pi)
		elif BFG.Name is Outflow:
			EqnSet.set_BC(BC_type="Extrapolate")


### Solve
solver = Solver.ADERDG_Solver(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, EqnSet, solver, "Scalar")
# Plot
Plot.PreparePlot()
Plot.PlotSolution(mesh, EqnSet, solver, "Scalar", PlotExact=True, PlotIC=True, Label="u")
Plot.ShowPlot()


# code.interact(local=locals())
