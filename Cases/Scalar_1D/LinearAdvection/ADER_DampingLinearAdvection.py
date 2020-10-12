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
mesh = MeshCommon.mesh_1D(Uniform=True, num_elems=16, xmin=-1., xmax=1., Periodic=Periodic)
# Non-uniform mesh
# num_elems = 25
# node_coords = np.cos(np.linspace(np.pi,0.,num_elems+1))
# node_coords = MeshCommon.refine_uniform_1D(node_coords)
# # node_coords = MeshCommon.refine_uniform_1D(node_coords)
# mesh = MeshCommon.mesh_1D(node_coords=node_coords, Periodic=Periodic)


### Solver parameters
FinalTime = 0.5
NumTimeSteps = np.amax([1,int(FinalTime/((mesh.node_coords[-1,0] - mesh.node_coords[-2,0])*0.1))])
SolutionOrder = 3
Params = general.set_solver_params(SolutionOrder=SolutionOrder,FinalTime=FinalTime,NumTimeSteps=NumTimeSteps,
								 SolutionBasis="LagrangeSeg",InterpolateFluxADER=True,SourceTreatmentADER="Implicit")
nu = -3.

### Physics
Velocity = 1.0 
physics = Scalar.ConstAdvScalar1D(Params["SolutionOrder"], Params["SolutionBasis"], mesh)
physics.set_physical_params(ConstVelocity=Velocity)
physics.set_conv_num_flux("LaxFriedrichs")

physics.set_IC(IC_type="Sine", omega = 2*np.pi)
physics.set_exact(exact_type="DampingSine", omega = 2*np.pi, nu = nu)
physics.set_source(source_type="SimpleSource", nu = nu)

# Boundary conditions
if Velocity >= 0.:
	Inflow = "x1"; Outflow = "x2"
else:
	Inflow = "x2"; Outflow = "x1"

# if not Periodic:
# 	for bgroup_num in range(mesh.num_boundary_groups):
# 		BFG = mesh.boundary_groups[bgroup_num]
# 		if BFG.Name is Inflow:
# 			physics.set_BC(BC_type="StateAll", fcn_type="DampingSine", omega = 2*np.pi, nu=nu)
# 		elif BFG.Name is Outflow:
# 			physics.set_BC(BC_type="Extrapolate")

if not Periodic:
	physics.set_BC(bname=Inflow, BC_type="StateAll", fcn_type="DampingSine", omega = 2*np.pi, nu=nu)
	physics.set_BC(bname=Outflow, BC_type="Extrapolate")


### Solve
solver = Solver.ADERDG(Params,physics,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.get_error(mesh, physics, solver, "Scalar")
# Plot
Plot.prepare_plot()
Plot.PlotSolution(mesh, physics, solver, "Scalar", PlotExact=True, PlotIC=True, Label="u")
Plot.show_plot()


# code.interact(local=locals())
