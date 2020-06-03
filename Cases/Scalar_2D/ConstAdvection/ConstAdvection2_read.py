import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import solver.DG as Solver
import physics.scalar.scalar as Scalar
import meshing.common as MeshCommon
import processing.post as Post
import processing.plot as Plot
import general
import meshing.gmsh as MeshGmsh
import os
import MeshTools
import processing.readwritedatafiles as ReadWriteDataFiles


### NOTE: STILL NEED TO BE ABLE TO RESTART WITH DIFFERENT BASIS/ORDER AND PROJECT


CurrentDir = os.path.dirname(os.path.abspath(__file__)) + "/"


fname = "Data1.pkl"
mesh, EqnSet, Params, _ = ReadWriteDataFiles.read_data_file(fname)

### Solver parameters
# InterpBasis = "LagrangeEqTri"
InterpBasis = "HierarchicH1Tri"
dt = 0.05
StartTime = 2.5
EndTime = 10.0
nTimeStep = int((EndTime-StartTime)/dt + 10.*general.eps)
InterpOrder = 10
Params = general.SetSolverParams(Params, InterpOrder=InterpOrder,StartTime=StartTime,EndTime=EndTime,
								 nTimeStep=nTimeStep,InterpBasis=InterpBasis,TimeScheme="RK4",
								 InterpolateIC=False,ApplyLimiter=None,WriteInterval=50,
								 RestartFile=fname)

### Solve
solver = Solver.DG_Solver(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
solver.Time = 0.
TotErr,_ = Post.L2_error(mesh, EqnSet, solver, "Scalar")
# Plot
axis = None
# axis = [-5., 5., -5., 5.]
Plot.PreparePlot(axis=axis, linewidth=0.5)
Plot.PlotSolution(mesh, EqnSet, solver, "Scalar", Equidistant=True, PlotExact=False, IncludeMesh2D=True, 
	Regular2D=True, ShowTriangulation=False)
Plot.SaveFigure(FileName=CurrentDir+'Gaussian', FileType='pdf', CropLevel=2)
Plot.plot_line_probe(mesh, EqnSet, solver, "Scalar", xy1=[-5.,-5.], xy2=[5.,5.], nPoint=101, PlotExact=True, PlotIC=True)
# Post.get_boundary_info(mesh, EqnSet, solver, "y1", "Scalar", integrate=True, 
# 		vec=[0.,1.], dot_normal_with_vec=True, plot_vs_x=True, plot_vs_y=False)
Plot.ShowPlot()

# U = EqnSet.U.Arrays[0]
# code.interact(local=locals())
