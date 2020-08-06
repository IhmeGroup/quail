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
mesh = MeshCommon.mesh_1D(Uniform=True, num_elems=50, xmin=0., xmax=1., Periodic=Periodic)
# Non-uniform mesh
# num_elems = 25
# node_coords = np.cos(np.linspace(np.pi,0.,num_elems+1))
# node_coords = MeshCommon.refine_uniform_1D(node_coords)
# # node_coords = MeshCommon.refine_uniform_1D(node_coords)
# mesh = MeshCommon.mesh_1D(node_coords=node_coords, Periodic=Periodic)


### Solver parameters
#dt = 0.001
mu = 1.
EndTime = 0.3
NumTimeSteps = np.amax([1,int(EndTime/((mesh.node_coords[1,0] - mesh.node_coords[0,0])*0.1))])
#NumTimeSteps = int(EndTime/dt)
InterpOrder = 2
Params = general.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,NumTimeSteps=NumTimeSteps,
								 InterpBasis="LagrangeSeg",TimeScheme="ADER")
								 #ApplyLimiter="ScalarPositivityPreserving")

### Physics
Velocity = 1.0
physics = Scalar.Burgers(Params["InterpOrder"], Params["InterpBasis"], mesh)
physics.set_physical_params(ConstVelocity=Velocity)
#physics.set_physical_params(AdvectionOperator="Burgers")
physics.set_physical_params(ConvFlux="LaxFriedrichs")


# ----------------------------------------------------------------------------------------------------
# This case is designed to test the time integration schemes ability to handle a stiff source term.
# In this function, 'SetSource', we manipulate the stiffness using a parameter that as it approaches
# zero, increases the amount of stiffness in the solution.
#
# For Example: If stiffness is set to 0.1, then the equation is not very stiff. But, if it is set to 
# something lower, (i.e. 0.001) then you will observe a stable solution, but the location of the shock
# will not be correct. It will have propogated at some other speed. (Note: RK4 cannot run stiff case)
# -----------------------------------------------------------------------------------------------------
physics.SetSource(Function=physics.FcnStiffSource,beta=0.5, stiffness = 1.)
# Initial conditions
physics.IC.Set(Function=physics.FcnScalarShock, uL = 1., uR = 0.,  xshock = 0.3)

# Exact solution
physics.ExactSoln.Set(Function=physics.FcnScalarShock, uL = 1., uR = 0.,  xshock = 0.3)
# Boundary conditions
if Velocity >= 0.:
	Inflow = "Left"; Outflow = "Right"
else:
	Inflow = "Right"; Outflow = "Left"
if not Periodic:
	for ibfgrp in range(mesh.nBFaceGroup):
		BC = physics.BCs[ibfgrp]
		## Left
		if BC.Name is Inflow:
			BC.Set(Function=physics.FcnScalarShock, BCType=physics.BCType["StateAll"], uL = 1., uR = 0., xshock = 0.3)
			#BC.Set(Function=physics.FcnUniform, BCType=physics.BCType["StateAll"], State = [1.])
		elif BC.Name is Outflow:
			BC.Set(BCType=physics.BCType["Extrapolation"])
			# BC.Set(Function=physics.FcnSine, BCType=physics.BCType["StateAll"], omega = 2*np.pi)
		else:
			raise Exception("BC error")


### Solve
solver = Solver.ADERDG(Params,physics,mesh)
solver.solve()


### Postprocess
# Error
#TotErr,_ = Post.L2_error(mesh, physics, solver.Time, "Scalar")
# Plot
Plot.PreparePlot()
Plot.PlotSolution(mesh, physics, solver.Time, "Scalar", PlotExact = True, PlotIC = True, Label="Q_h")
Plot.ShowPlot()


# code.interact(local=locals())
