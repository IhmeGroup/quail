import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import solver.ADERDG as Solver
import physics.scalar.scalar as Scalar
import meshing.common as MeshCommon
import processing.post as Post
import processing.plot as Plot
import general
import meshing.tools as MeshTools


### Mesh
Periodic = False 
# Uniform mesh
mesh = MeshCommon.mesh_1D(num_elems=32, xmin=-1., xmax=1.)
if Periodic:
	MeshTools.make_periodic_translational(mesh, x1="x1", x2="x2")
# Non-uniform mesh
# num_elems = 25
# node_coords = np.cos(np.linspace(np.pi,0.,num_elems+1))
# node_coords = MeshCommon.refine_uniform_1D(node_coords)
# # node_coords = MeshCommon.refine_uniform_1D(node_coords)
# mesh = MeshCommon.mesh_1D(node_coords=node_coords, Periodic=Periodic)


### Solver parameters
FinalTime = 0.1
NumTimeSteps = np.amax([1,int(FinalTime/((mesh.node_coords[-1,0] - mesh.node_coords[-2,0])*0.1))])
SolutionOrder = 2
Params = general.SetSolverParams(SolutionOrder=SolutionOrder,FinalTime=FinalTime,NumTimeSteps=NumTimeSteps,
								 SolutionBasis="LagrangeSeg",TimeStepper="ADER")


### Physics
Velocity = 1.0
physics = Scalar.ConstAdvScalar1D(Params["SolutionOrder"], Params["SolutionBasis"], mesh)
physics.set_physical_params(ConstVelocity=Velocity)
#physics.set_physical_params(AdvectionOperator="Burgers")
physics.set_conv_num_flux("LaxFriedrichs")
# Initial conditions
# physics.IC.Set(Function=physics.FcnSine, omega = 2*np.pi)
physics.set_IC(IC_type="Sine", omega = 2*np.pi)
# Exact solution
# physics.ExactSoln.Set(Function=physics.FcnSine, omega = 2*np.pi)
physics.set_exact(exact_type="Sine", omega = 2*np.pi)
# Boundary conditions
if Velocity >= 0.:
	Inflow = "x1"; Outflow = "x2"
else:
	Inflow = "x2"; Outflow = "x1"
# if not Periodic:
# 	for ibfgrp in range(mesh.num_boundary_groups):
# 		BFG = mesh.boundary_groups[ibfgrp]
# 		if BFG.Name is Inflow:
# 			physics.set_BC(BC_type="StateAll", fcn_type="Sine", omega = 2*np.pi)
# 		elif BFG.Name is Outflow:
# 			physics.set_BC(BC_type="Extrapolate")

if not Periodic:
	physics.set_BC(bname=Inflow, BC_type="StateAll", fcn_type="Sine", omega = 2*np.pi)
	physics.set_BC(bname=Outflow, BC_type="Extrapolate")

### Solve
solver = Solver.ADERDG(Params,physics,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.get_error(mesh, physics, solver, "Scalar")
# Plot
# Plot.prepare_plot()
# Plot.PlotSolution(mesh, physics, solver, "Scalar", PlotExact=True, PlotIC=True, Label="u")
# Plot.show_plot()


# code.interact(local=locals())
