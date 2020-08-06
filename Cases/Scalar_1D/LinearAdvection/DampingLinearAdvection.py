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
EndTime = 0.5
NumTimeSteps = np.amax([1,int(EndTime/((mesh.node_coords[1,0] - mesh.node_coords[0,0])*0.1))])
InterpOrder = 2
Params = general.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,NumTimeSteps=NumTimeSteps,
								 InterpBasis="LagrangeSeg",InterpolateFlux=True)
nu = -3.

### Physics
Velocity = 1.0 
physics = Scalar.ConstAdvScalar1D(Params["InterpOrder"], Params["InterpBasis"], mesh)
physics.set_physical_params(ConstVelocity=Velocity)
physics.set_conv_num_flux("LaxFriedrichs")

physics.set_IC(IC_type="Sine", omega = 2*np.pi)
physics.set_exact(exact_type="DampingSine", omega = 2*np.pi, nu = nu)
physics.set_source(source_type="SimpleSource", nu = nu)




# Boundary conditions
if Velocity >= 0.:
	Inflow = "Left"; Outflow = "Right"
else:
	Inflow = "Right"; Outflow = "Left"
# if not Periodic:
# 	for ibfgrp in range(mesh.nBFaceGroup):
# 		BFG = mesh.BFaceGroups[ibfgrp]
# 		if BFG.Name is Inflow:
# 			physics.set_BC(BC_type="StateAll", fcn_type="DampingSine", omega = 2*np.pi, nu=nu)
# 		elif BFG.Name is Outflow:
# 			physics.set_BC(BC_type="Extrapolate")

if not Periodic:
	physics.set_BC(bname=Inflow, BC_type="StateAll", fcn_type="DampingSine", omega = 2*np.pi, nu=nu)
	physics.set_BC(bname=Outflow, BC_type="Extrapolate")

### Solve
solver = Solver.DG(Params,physics,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, physics, solver, "Scalar")
# Plot
# Plot.PreparePlot()
# Plot.PlotSolution(mesh, physics, solver, "Scalar", PlotExact=True, PlotIC=True, Label="u")
# Plot.ShowPlot()


# code.interact(local=locals())
