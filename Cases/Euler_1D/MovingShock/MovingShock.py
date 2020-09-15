import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import solver.DG as Solver
import physics.euler.euler as Euler
import meshing.common as MeshCommon
import processing.post as Post
import processing.plot as Plot
import general
import os
import meshing.tools as MeshTools


CurrentDir = os.path.dirname(os.path.abspath(__file__)) + "/"


### Mesh
mesh = MeshCommon.mesh_1D(num_elems=100, xmin=0., xmax=1.)


### Solver parameters
FinalTime = 4.e-5
NumTimeSteps = 100
SolutionOrder = 1

Params = general.set_solver_params(SolutionOrder=SolutionOrder,FinalTime=FinalTime,NumTimeSteps=NumTimeSteps,
								 SolutionBasis="LagrangeSeg",TimeStepper="SSPRK3",L2InitialCondition=True,
								 ApplyLimiter="PositivityPreserving")


### Physics
physics = Euler.Euler1D(Params["SolutionOrder"], Params["SolutionBasis"], mesh)
physics.set_physical_params(GasConstant=287.,SpecificHeatRatio=1.4)
physics.set_conv_num_flux("Roe")
# Parameters
M = 5.
xshock = 0.2
# Initial conditions
physics.set_IC(IC_type="MovingShock", M=M, xshock=xshock)
# Exact solution
physics.set_exact(exact_type="MovingShock", M=M, xshock = xshock)

# Boundary conditions
# for ibfgrp in range(mesh.num_boundary_groups):
# 	BFG = mesh.boundary_groups[ibfgrp]
# 	if BFG.Name is "x1":
# 		physics.set_BC(BC_type="StateAll", fcn_type="MovingShock", M=M, xshock=xshock)
# 	elif BFG.Name is "x2":
# 		physics.set_BC(BC_type="StateAll", fcn_type="MovingShock", M=M, xshock=xshock)

physics.set_BC(bname="x1", BC_type="StateAll", fcn_type="MovingShock", M=M, xshock=xshock)
physics.set_BC(bname="x2", BC_type="StateAll", fcn_type="MovingShock", M=M, xshock=xshock)

# physics.set_BC(BC_type="StateAll", fcn_type="MovingShock")
# physics.SetBC("x2",Function=physics.FcnMovingShock, BCType=physics.BCType["StateAll"], M=M, xshock=xshock)


### Solve
solver = Solver.DG(Params,physics,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.get_error(mesh, physics, solver, "Density")
# Plot
# Plot.prepare_plot()
# Plot.PlotSolution(mesh, physics, solver, "Pressure", PlotExact=True, Equidistant=True)
# Plot.save_figure(file_name=CurrentDir+'Pressure', file_type='pdf', crop_level=2)
# Plot.show_plot()


# code.interact(local=locals())
