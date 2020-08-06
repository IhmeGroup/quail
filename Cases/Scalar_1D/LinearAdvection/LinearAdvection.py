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
mesh = MeshCommon.mesh_1D(Uniform=True, num_elems=32, xmin=-1., xmax=1., Periodic=Periodic)
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
								 InterpBasis="LegendreSeg",TimeScheme="LSRK4")


### Physics
Velocity = 1.
physics = Scalar.ConstAdvScalar1D(Params["InterpOrder"], Params["InterpBasis"], mesh)
# physics.set_physical_params(ConstVelocity=Velocity)
physics.set_physical_params(ConstVelocity=Velocity)
# Initial conditions
# physics.IC.Set(Function=physics.FcnSine, omega = 2*np.pi)
physics.set_conv_num_flux(conv_num_flux_type="LaxFriedrichs")
physics.set_IC(IC_type="Sine", omega = 2*np.pi)
# Exact solution
# physics.ExactSoln.Se”t(Function=physics.FcnSine, omega = 2*np.pi)
physics.set_exact(exact_type="Sine", omega = 2*np.pi)
# Boundary conditions
if Velocity >= 0.:
	Inflow = "Left"; Outflow = "Right"
else:
	Inflow = "Right"; Outflow = "Left"
if not Periodic:
	raise Exception
	# for ibfgrp in range(mesh.num_boundary_groups):
	# 	BC = physics.BCs[ibfgrp]
	# 	## Left
	# 	if BC.Name is Inflow:
	# 		BC.Set(Function=physics.FcnSine, BCType=physics.BCType["StateAll"], omega = 2*np.pi)
	# 	elif BC.Name is Outflow:
	# 		BC.Set(BCType=physics.BCType["Extrapolation"])
	# 		# BC.Set(Function=physics.FcnSine, BCType=physics.BCType["StateAll"], omega = 2*np.pi)
	# 	else:
	# 		raise Exception("BC error")


### Solve
solver = Solver.DG(Params,physics,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, physics, solver, "Scalar")
# Plot
# Plot.PreparePlot()
# Plot.PlotSolution(mesh, physics, solver, "Scalar", PlotExact=True, Label="Q_h", include_mesh=True, EqualAR=True, show_elem_IDs=True)
# Plot.ShowPlot()


# code.interact(local=locals())
