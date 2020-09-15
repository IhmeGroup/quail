import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import solver.ADERDG as Solver
import physics.euler.euler as Euler
import meshing.common as MeshCommon
import processing.post as Post
import processing.plot as Plot
import general
import meshing.tools as MeshTools


### Mesh
Periodic = False
mesh = MeshCommon.mesh_1D(num_elems=25, xmin=-1., xmax=1.)


### Solver parameters
FinalTime = 0.1
NumTimeSteps = 100
SolutionOrder = 2
Params = general.set_solver_params(SolutionOrder=SolutionOrder,FinalTime=FinalTime,NumTimeSteps=NumTimeSteps,
								 SolutionBasis="LagrangeSeg",TimeStepper="ADER",L2InitialCondition=False,ApplyLimiter=None)

# nu = -1000.
### Physics
physics = Euler.Euler1D(Params["SolutionOrder"], Params["SolutionBasis"], mesh)
physics.set_physical_params(GasConstant=1.,SpecificHeatRatio=3.)
physics.set_conv_num_flux("LaxFriedrichs")
# Initial conditions
physics.set_IC(IC_type="SmoothIsentropicFlow", a=0.9)
physics.set_exact(exact_type="SmoothIsentropicFlow", a=0.9)

# Boundary conditions
# if not Periodic:
# 	for ibfgrp in range(mesh.num_boundary_groups):
# 		BFG = mesh.boundary_groups[ibfgrp]
# 		if BFG.Name is "x1":
# 			physics.set_BC(BC_type="StateAll", fcn_type="SmoothIsentropicFlow", a=0.9)
# 		elif BFG.Name is "x2":
# 			physics.set_BC(BC_type="StateAll", fcn_type="SmoothIsentropicFlow", a=0.9)
if not Periodic:
	physics.set_BC(bname="x1", BC_type="StateAll", fcn_type="SmoothIsentropicFlow", a=0.9)
	physics.set_BC(bname="x2", BC_type="StateAll", fcn_type="SmoothIsentropicFlow", a=0.9)

### Solve
solver = Solver.ADERDG(Params,physics,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.get_error(mesh, physics, solver, "Density")
# Plot
# Plot.prepare_plot()
# Plot.PlotSolution(mesh, physics, solver, "Energy", PlotExact=True, Equidistant=True)
# Plot.show_plot()


# code.interact(local=locals())
