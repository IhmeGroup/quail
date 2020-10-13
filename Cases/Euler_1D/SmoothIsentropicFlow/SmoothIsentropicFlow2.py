import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import solver.DG as Solver
import physics.euler.euler as Euler
import meshing.common as MeshCommon
import processing.post as Post
import processing.plot as Plot
import general
import meshing.gmsh as MeshGmsh
import meshing.tools as MeshTools


### Mesh
Periodic = False
# mesh = MeshCommon.mesh_1D(Uniform=True, num_elems=2, xmin=-1., xmax=1., Periodic=Periodic)
mesh = MeshGmsh.import_gmsh_mesh("mesh_v2.msh")
# code.interact(local=locals())

# Test "random" 1D mesh
# mesh.elem_to_node_IDs[1,:] = np.array([1,2])
# MeshTools.RandomizeNodes(mesh)
mesh.interior_faces[0].elemL_ID = 1
mesh.interior_faces[0].elemR_ID = 0
mesh.interior_faces[0].faceL_ID = 0
mesh.interior_faces[0].faceR_ID = 1
# code.interact(local=locals())

### Solver parameters
FinalTime = 0.1
NumTimeSteps = 100
SolutionOrder = 5

Params = general.set_solver_params(SolutionOrder=SolutionOrder,FinalTime=FinalTime,NumTimeSteps=NumTimeSteps,
								 SolutionBasis="LagrangeSeg",TimeStepper="RK4",L2InitialCondition=False)


### Physics
physics = Euler.Euler1D(mesh)
physics.set_physical_params(GasConstant=1.,SpecificHeatRatio=3.)
physics.set_conv_num_flux("LaxFriedrichs")
# Initial conditions
physics.set_IC(IC_type="SmoothIsentropicFlow", a=0.9)
physics.set_exact(exact_type="SmoothIsentropicFlow", a=0.9)

# Boundary conditions
# if not Periodic:
# 	for bgroup_num in range(mesh.num_boundary_groups):
# 		BFG = mesh.boundary_groups[bgroup_num]
# 		if BFG.Name is "x1":
# 			physics.set_BC(BC_type="StateAll", fcn_type="SmoothIsentropicFlow", a=0.9)
# 		elif BFG.Name is "x2":
# 			physics.set_BC(BC_type="StateAll", fcn_type="SmoothIsentropicFlow", a=0.9)
if not Periodic:
	physics.set_BC(bname="x1", BC_type="StateAll", fcn_type="SmoothIsentropicFlow", a=0.9)
	physics.set_BC(bname="x2", BC_type="StateAll", fcn_type="SmoothIsentropicFlow", a=0.9)

### Solve
solver = Solver.DG(Params,physics,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.get_error(mesh, physics, solver, "Density")
# Plot
# Plot.prepare_plot()
# Plot.PlotSolution(mesh, physics, solver, "Energy", PlotExact=True, Equidistant=True, include_mesh=True, equal_AR=True, show_elem_IDs=True)
# Plot.show_plot()


# code.interact(local=locals())
