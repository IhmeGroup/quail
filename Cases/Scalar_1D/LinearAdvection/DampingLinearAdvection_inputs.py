import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import processing.post as Post
import processing.plot as Plot
import os
import driver
import general
import code


### Mesh
# Periodic = False
# # Uniform mesh
# mesh = MeshCommon.mesh_1D(Uniform=True, num_elems=16, xmin=-1., xmax=1., Periodic=Periodic)
# # Non-uniform mesh
# # num_elems = 25
# # node_coords = np.cos(np.linspace(np.pi,0.,num_elems+1))
# # node_coords = MeshCommon.refine_uniform_1D(node_coords)
# # # node_coords = MeshCommon.refine_uniform_1D(node_coords)
# # mesh = MeshCommon.mesh_1D(node_coords=node_coords, Periodic=Periodic)


# ### Solver parameters
# EndTime = 0.5
# num_time_steps = np.amax([1,int(EndTime/((mesh.node_coords[-1,0] - mesh.node_coords[-2,0])*0.1))])
# InterpOrder = 2
# Params = general.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,num_time_steps=num_time_steps,
# 								 InterpBasis="LagrangeEqSeg",TimeScheme="RK4",InterpolateFlux=True)
# nu = -3.

# ### Physics
# Velocity = 1.0 
# physics = Scalar.ConstAdvScalar1D(Params["InterpOrder"], Params["InterpBasis"], mesh)
# physics.set_physical_params(ConstVelocity=Velocity)
# physics.set_physical_params(ConvFlux="LaxFriedrichs")
# physics.SetSource(Function=physics.FcnSimpleSource, nu = nu)


# Uinflow=[1.]
# # Initial conditions
# physics.IC.Set(Function=physics.FcnDampingSine, omega = 2.*np.pi , nu = nu)
# # Exact solution
# physics.ExactSoln.Set(Function=physics.FcnDampingSine, omega = 2.*np.pi , nu = nu)
# # Boundary conditions
# if Velocity >= 0.:
# 	Inflow = "Left"; Outflow = "Right"
# else:
# 	Inflow = "Right"; Outflow = "Left"
# if not Periodic:
# 	for ibfgrp in range(mesh.num_boundary_groups):
# 		BC = physics.BCs[ibfgrp]
# 		## Left
# 		if BC.Name is Inflow:
# 			BC.Set(Function=physics.FcnDampingSine, BCType=physics.BCType["StateAll"], omega = 2.*np.pi, nu=nu)
# 		elif BC.Name is Outflow:
# 			BC.Set(BCType=physics.BCType["Extrapolation"])
# 			#BC.Set(Function=physics.FcnDampingSine, BCType=physics.BCType["StateAll"], omega = 2*np.pi, nu=-2.0)
# 		else:
# 			raise Exception("BC error")


# ### Solve
# solver = Solver.DG(Params,physics,mesh)
# solver.solve()



TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : 0.5,
    "num_time_steps" : 40,
    "TimeScheme" : "RK4",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeEqSeg",
    "Solver" : "DG",
}

Output = {}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 16,
    "NumElemsY" : 2,
    "xmin" : -1.,
    "xmax" : 1.,
    # "PeriodicBoundariesX" : ["x1", "x2"],
    # "PeriodicBoundariesY" : ["y1", "y2"],
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFluxNumerical" : "LaxFriedrichs",
    "ConstVelocity" : 1.,
}

nu = -3.
InitialCondition = {
    "Function" : "DampingSine",
    "omega" : 2*np.pi,
    "nu" : nu,
    "SetAsExact" : True,
}

BoundaryConditions = {
    "Left" : {
	    "Function" : "DampingSine",
	    "omega" : 2*np.pi,
	    "nu" : nu,
    	"BCType" : "StateAll",
    },
    "Right" : {
    	"Function" : None,
    	"BCType" : "Extrapolation",
    },
}

SourceTerms = {
	"source1" : {
		"Function" : "SimpleSource",
		"nu" : nu,
	},
}


solver, physics, mesh = driver.driver(TimeStepping, Numerics, Output, Mesh,
		Physics, InitialCondition, BoundaryConditions, SourceTerms)


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, physics, solver, "Scalar")
# Plot
Plot.PreparePlot()
Plot.PlotSolution(mesh, physics, solver, "Scalar", PlotExact=True, PlotIC=True, Label="u")
Plot.ShowPlot()


# code.interact(local=locals())
