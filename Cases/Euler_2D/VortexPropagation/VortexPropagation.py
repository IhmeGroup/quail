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
import os
# import meshing.tools as MeshTools


CurrentDir = os.path.dirname(os.path.abspath(__file__)) + "/"


### Mesh
folder = "meshes/"
# Quadrilaterals
#subfolder = "Quadrilaterals/"; InterpBasis = "LagrangeEqQuad"
# # # Structured
#subsubfolder = "Structured/"
#FileName = "box_5x5.msh"
# FileName = "box_10x10.msh"
# FileName = "box_20x20.msh"
# FileName = "box_40x40.msh"
# FileName = "box_80x80.msh"
# Unstructured
#subsubfolder = "Unstructured/"
#FileName = "box_25_elem.msh"
# FileName = "box_100_elem.msh"
# FileName = "box_400_elem.msh"
# FileName = "box_1600_elem.msh"
## Triangles
subfolder = "Triangles/"; InterpBasis = "LagrangeTri"
# Structured
subsubfolder = "Structured/"
FileName = "box_5x5_v4.msh"
# FileName = "box_10x10.msh"
# FileName = "box_20x20.msh"
# FileName = "box_40x40.msh"
MeshFile = os.path.dirname(os.path.abspath(__file__)) + "/" + folder + subfolder + subsubfolder + FileName
mesh = MeshGmsh.import_gmsh_mesh(MeshFile)

# MeshTools.RandomizeNodes(mesh)

### Solver parameters
dt = 0.05
EndTime = 1.0
NumTimeSteps = int(EndTime/dt + 10.*general.eps)
InterpOrder = 2
Params = general.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,NumTimeSteps=NumTimeSteps,
								 InterpBasis=InterpBasis,TimeScheme="RK4",InterpolateIC=False,
								 ElementQuadrature="Dunavant")


### Physics
physics = Euler.Euler2D(Params["InterpOrder"], Params["InterpBasis"], mesh)
# physics.set_physical_params(GasConstant=1.,SpecificHeatRatio=1.4)
physics.set_physical_params(GasConstant=1.,SpecificHeatRatio=1.4)
physics.set_conv_num_flux("LaxFriedrichs")
# Initial conditions
physics.set_IC(IC_type="IsentropicVortex")
# Exact solution
physics.set_exact(exact_type="IsentropicVortex")
# physics.ExactSoln.Set(Function=physics.FcnIsentropicVortexPropagation)
# Boundary conditions
physics.set_BC(bname="wall", BC_type="StateAll", fcn_type="IsentropicVortex")
# physics.SetBC("wall",Function=physics.FcnIsentropicVortexPropagation, BCType=physics.BCType["StateAll"])
# raise Exception


### Solve
solver = Solver.DG(Params,physics,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, physics, solver, "Density")
# Plot
axis = None
# axis = [-5., 5., -5., 5.]
# Plot.PreparePlot(axis=axis, linewidth=0.5)
# Plot.PlotSolution(mesh, physics, solver, "Density", Equidistant=True, PlotExact=False, include_mesh=True, 
# 	Regular2D=True, show_triangulation=False, show_elem_IDs=True)
# Plot.SaveFigure(FileName=CurrentDir+'vortex', FileType='pdf', CropLevel=2)
# Plot.PreparePlot(close_all=False, linewidth=1.5)
# Plot.plot_line_probe(mesh, physics, solver, "Density", xy1=[-5.,1.], xy2=[5.,1.], nPoint=101, PlotExact=True, PlotIC=True)
# Plot.SaveFigure(FileName=CurrentDir+'line', FileType='pdf', CropLevel=2)
# Plot.ShowPlot()

# U = physics.U.Arrays[0]
# code.interact(local=locals())
