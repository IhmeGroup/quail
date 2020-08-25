import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import solver.DG as Solver
import physics.scalar.scalar as Scalar
import meshing.common as MeshCommon
import processing.post as Post
import processing.plot as Plot
import general
import meshing.tools as MeshTools


### Mesh
Periodic = True
# Uniform mesh
mesh = MeshCommon.mesh_1D(num_elems=16, xmin=-1., xmax=1.)
if Periodic:
	MeshTools.make_periodic_translational(mesh, x1="x1", x2="x2")
# Non-uniform mesh
# num_elems = 25
# node_coords = np.cos(np.linspace(np.pi,0.,num_elems+1))
# node_coords = MeshCommon.refine_uniform_1D(node_coords)
# # node_coords = MeshCommon.refine_uniform_1D(node_coords)
# mesh = MeshCommon.mesh_1D(node_coords=node_coords, Periodic=Periodic)


### Solver parameters
#dt = 0.001
#mu = 1.
FinalTime = 0.1
NumTimeSteps = np.amax([1,int(FinalTime/((mesh.node_coords[-1,0] - mesh.node_coords[-2,0])*0.1))])
#NumTimeSteps = int(FinalTime/dt)
SolutionOrder = 3
Params = general.SetSolverParams(SolutionOrder=SolutionOrder,FinalTime=FinalTime,NumTimeSteps=NumTimeSteps,
								 SolutionBasis="LagrangeSeg",TimeStepper="RK4")
### Physics
ConstVelocity = 1.
physics = Scalar.Burgers1D(Params["SolutionOrder"], Params["SolutionBasis"], mesh)
#physics.set_physical_params(AdvectionOperator="Burgers")
# physics.set_physical_params(ConstVelocity=ConstVelocity)
physics.set_conv_num_flux("LaxFriedrichs")

# Initial conditions
# physics.IC.Set(Function=physics.FcnSine, omega = 2*np.pi)
physics.set_IC(IC_type="SineBurgers", omega = 2*np.pi)
# Exact solution
# physics.ExactSoln.Set(Function=physics.FcnSineWaveBurgers, omega = 2*np.pi)
physics.set_exact(exact_type="SineBurgers", omega = 2*np.pi)
# Boundary conditions
if ConstVelocity >= 0.:
	Inflow = "x1"; Outflow = "x2"
else:
	Inflow = "x2"; Outflow = "x1"
if not Periodic:
	for ibfgrp in range(mesh.num_boundary_groups):
		BC = physics.BCs[ibfgrp]
		## Left
		if BC.Name is Inflow:
			BC.Set(Function=physics.FcnSine, BCType=physics.BCType["StateAll"])
		elif BC.Name is Outflow:
			BC.Set(BCType=physics.BCType["Extrapolation"])
			# BC.Set(Function=physics.FcnSine, BCType=physics.BCType["StateAll"], omega = 2*np.pi)
		else:
			raise Exception("BC error")


### Solve
solver = Solver.DG(Params,physics,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.get_error(mesh, physics, solver, "Scalar")
# Plot
# Plot.prepare_plot()
# Plot.PlotSolution(mesh, physics, solver, "Scalar", PlotExact = True, PlotIC = True, Label="u")
# Plot.show_plot()


# code.interact(local=locals())
