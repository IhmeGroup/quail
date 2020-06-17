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
# nElem_x = 2
# mesh = MeshCommon.mesh_2D(xcoords=None, ycoords=None, nElem_x= nElem_x, nElem_y = nElem_x, Uniform=True, xmin=-5., xmax=5., 
# 	ymin=-5., ymax=5., Periodic=Periodic)
# if Periodic:
# 	MeshTools.MakePeriodicTranslational(mesh, x1="x1", x2="x2", y1="y1", y2="y2")

# ### Solver parameters
# # InterpBasis = "LagrangeEqTri"
# InterpBasis = "HierarchicH1Tri"
# if InterpBasis is "LagrangeEqTri" or "HierarchicH1Tri":
# 	mesh = MeshCommon.split_quadrils_into_tris(mesh)
# dt = 0.05
# EndTime = 10.0
# nTimeStep = int(EndTime/dt + 10.*general.eps)
# InterpOrder = 10
# Params = general.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
# 								 InterpBasis=InterpBasis,TimeScheme="RK4",InterpolateIC=False,
# 								 ApplyLimiter=None,WriteInterval=50)


# ### Physics
# x0 = np.array([0., 0.])
# EqnSet = Scalar.ConstAdvScalar2D(Params["InterpOrder"], Params["InterpBasis"], mesh)
# EqnSet.set_physical_params(ConstXVelocity=1.,ConstYVelocity=1.,ConvFlux="LaxFriedrichs")
# # Initial conditions
# EqnSet.IC.Set(Function=EqnSet.FcnGaussian, x0=x0)
# # Exact solution
# EqnSet.ExactSoln.Set(Function=EqnSet.FcnGaussian, x0=x0)
# # Boundary conditions
# if not Periodic:
# 	MeshTools.check_face_orientations(mesh)
# 	EqnSet.SetBC("x1",Function=EqnSet.FcnGaussian, x0=x0, BCType=EqnSet.BCType["StateAll"])
# 	EqnSet.SetBC("x2",Function=EqnSet.FcnGaussian, x0=x0, BCType=EqnSet.BCType["StateAll"])
# 	EqnSet.SetBC("y1",Function=EqnSet.FcnGaussian, x0=x0, BCType=EqnSet.BCType["StateAll"])
# 	EqnSet.SetBC("y2",Function=EqnSet.FcnGaussian, x0=x0, BCType=EqnSet.BCType["StateAll"])
# # raise Exception


# ### Solve
# solver = Solver.DG(Params,EqnSet,mesh)
# solver.solve()


TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : 10.,
    "nTimeStep" : 200,
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
    "nElem_x" : 2,
    "nElem_y" : 2,
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


solver, EqnSet, mesh = driver.driver(TimeStepping, Numerics, Output, Mesh,
		Physics, InitialCondition, BoundaryConditions, SourceTerms)


### Postprocess
# Error
solver.Time = 0.
TotErr,_ = Post.L2_error(mesh, EqnSet, solver, "Scalar")
# Plot
axis = None
# axis = [-5., 5., -5., 5.]
Plot.PreparePlot(axis=axis, linewidth=0.5)
Plot.PlotSolution(mesh, EqnSet, solver, "Scalar", Equidistant=True, PlotExact=False, include_mesh=True, 
	Regular2D=True, ShowTriangulation=False)
Plot.SaveFigure(FileName=CurrentDir+'Gaussian', FileType='pdf', CropLevel=2)
Plot.plot_line_probe(mesh, EqnSet, solver, "Scalar", xy1=[-5.,-5.], xy2=[5.,5.], nPoint=101, PlotExact=True, PlotIC=True)
# Post.get_boundary_info(mesh, EqnSet, solver, "y1", "Scalar", integrate=True, 
# 		vec=[0.,1.], dot_normal_with_vec=True, plot_vs_x=True, plot_vs_y=False)
Plot.ShowPlot()

# U = EqnSet.U.Arrays[0]
# code.interact(local=locals())
