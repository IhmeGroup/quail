import sys; sys.path.append('../../../src'); sys.path.append('./src')
# sys.path.append('../../../src/numerics/quadrature');sys.path.append('./src/numerics/quadrature')
# sys.path.append('../../../src/numerics/basis');sys.path.append('./src/numerics/basis')
import numpy as np
import processing.post as Post
import processing.plot as Plot
import os
import driver
import general
import code


CurrentDir = os.path.dirname(os.path.abspath(__file__)) + "/"

# Periodic = True
# num_elems_x = 2
# mesh = MeshCommon.mesh_2D(xcoords=None, ycoords=None, num_elems_x= num_elems_x, num_elems_y = num_elems_x, Uniform=True, xmin=-5., xmax=5., 
# 	ymin=-5., ymax=5., Periodic=Periodic)
# if Periodic:
# 	MeshTools.make_periodic_translational(mesh, x1="x1", x2="x2", y1="y1", y2="y2")

# ### Solver parameters
# # InterpBasis = "LagrangeEqTri"
# InterpBasis = "HierarchicH1Tri"
# if InterpBasis is "LagrangeEqTri" or "HierarchicH1Tri":
# 	mesh = MeshCommon.split_quadrils_into_tris(mesh)
# dt = 0.05
# EndTime = 10.0
# num_time_steps = int(EndTime/dt + 10.*general.eps)
# InterpOrder = 10
# Params = general.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,num_time_steps=num_time_steps,
# 								 InterpBasis=InterpBasis,TimeScheme="RK4",InterpolateIC=False,
# 								 ApplyLimiter=None,WriteInterval=50)


# ### Physics
# x0 = np.array([0., 0.])
# physics = Scalar.ConstAdvScalar2D(Params["InterpOrder"], Params["InterpBasis"], mesh)
# physics.set_physical_params(ConstXVelocity=1.,ConstYVelocity=1.,ConvFlux="LaxFriedrichs")
# # Initial conditions
# physics.IC.Set(Function=physics.FcnGaussian, x0=x0)
# # Exact solution
# physics.ExactSoln.Set(Function=physics.FcnGaussian, x0=x0)
# # Boundary conditions
# if not Periodic:
# 	MeshTools.check_face_orientations(mesh)
# 	physics.SetBC("x1",Function=physics.FcnGaussian, x0=x0, BCType=physics.BCType["StateAll"])
# 	physics.SetBC("x2",Function=physics.FcnGaussian, x0=x0, BCType=physics.BCType["StateAll"])
# 	physics.SetBC("y1",Function=physics.FcnGaussian, x0=x0, BCType=physics.BCType["StateAll"])
# 	physics.SetBC("y2",Function=physics.FcnGaussian, x0=x0, BCType=physics.BCType["StateAll"])
# # raise Exception


# ### Solve
# solver = Solver.DG(Params,physics,mesh)
# solver.solve()


TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : 10.,
    "num_time_steps" : 200,
    "TimeScheme" : "RK4",
}

Numerics = {
    "InterpOrder" : 10,
    "InterpBasis" : "HierarchicH1Tri",
    "Solver" : "DG",
    "InterpolateIC" : False,
    "InterpolateFlux" : False,
    "OrderSequencing" : False,
    "ApplyLimiter" : None, 
}

Output = {
    "TrackOutput" : None,
    "WriteTimeHistory" : False,
    "Prefix" : "Data",
    "WriteInterval" : 50,
    "WriteInitialSolution" : False,
    "WriteFinalSolution" : False,
    "RestartFile" : None,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Triangle",
    "NumElems_x" : 2,
    "NumElems_y" : 2,
    "xmin" : -5.,
    "xmax" : 5.,
    "ymin" : -5.,
    "ymax" : 5.,
    # "PeriodicBoundariesX" : ["x1", "x2"],
    # "PeriodicBoundariesY" : ["y1", "y2"],
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFlux" : "LaxFriedrichs",
    "ConstXVelocity" : 1.,
    "ConstYVelocity" : 1.,
}

x0 = [0., 0.]
InitialCondition = {
    "Function" : "Gaussian",
    "x0" : x0,
    "SetAsExact" : True,
}

bparams = {
    "Function" : "Gaussian",
    "x0" : x0,
    "BCType" : "StateAll"
}
BoundaryConditions = {
    "x1" : bparams,
    "x2" : bparams,
    "y1" : bparams,
    "y2" : bparams,
}
# BoundaryConditions = {}

SourceTerms = {}


solver, physics, mesh = driver.driver(TimeStepping, Numerics, Output, Mesh,
		Physics, InitialCondition, BoundaryConditions, SourceTerms)


### Postprocess
# Error
solver.time = 0.
TotErr,_ = Post.L2_error(mesh, physics, solver, "Scalar")
# Plot
axis = None
# axis = [-5., 5., -5., 5.]
Plot.PreparePlot(axis=axis, linewidth=0.5)
Plot.PlotSolution(mesh, physics, solver, "Scalar", Equidistant=True, PlotExact=False, include_mesh=True, 
	Regular2D=True, show_triangulation=False)
Plot.SaveFigure(FileName=CurrentDir+'Gaussian', FileType='pdf', CropLevel=2)
Plot.plot_line_probe(mesh, physics, solver, "Scalar", xy1=[-5.,-5.], xy2=[5.,5.], nPoint=101, PlotExact=True, PlotIC=True)
# Post.get_boundary_info(mesh, physics, solver, "y1", "Scalar", integrate=True, 
# 		vec=[0.,1.], dot_normal_with_vec=True, plot_vs_x=True, plot_vs_y=False)
Plot.ShowPlot()

# U = physics.U.Arrays[0]
# code.interact(local=locals())
