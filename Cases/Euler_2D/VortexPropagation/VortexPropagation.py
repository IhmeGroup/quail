import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import Solver
import physics.euler.euler as Euler
import meshing.common as MeshCommon
import Post
import Plot
import General
import meshing.gmsh as MeshGmsh
import os


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
subfolder = "Triangles/"; InterpBasis = "LagrangeEqTri"
# Structured
subsubfolder = "Structured/"
FileName = "box_5x5.msh"
# FileName = "box_10x10.msh"
# FileName = "box_20x20.msh"
# FileName = "box_40x40.msh"
MeshFile = os.path.dirname(os.path.abspath(__file__)) + "/" + folder + subfolder + subsubfolder + FileName
mesh = MeshGmsh.ReadGmshFile(MeshFile)

### Solver parameters
dt = 0.05
EndTime = 1.0
nTimeStep = int(EndTime/dt + 10.*General.eps)
InterpOrder = 2
Params = General.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
								 InterpBasis=InterpBasis,TimeScheme="RK4",InterpolateIC=False)


### Physics
EqnSet = Euler.Euler2D(Params["InterpOrder"], Params["InterpBasis"], mesh, StateRank=4)
EqnSet.SetParams(GasConstant=1.,SpecificHeatRatio=1.4,ConvFlux="LaxFriedrichs")
# Initial conditions
EqnSet.IC.Set(Function=EqnSet.FcnIsentropicVortexPropagation)
# Exact solution
EqnSet.ExactSoln.Set(Function=EqnSet.FcnIsentropicVortexPropagation)
# Boundary conditions
EqnSet.SetBC("wall",Function=EqnSet.FcnIsentropicVortexPropagation, BCType=EqnSet.BCType["FullState"])
# raise Exception


### Solve
solver = Solver.DG_Solver(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, EqnSet, solver, "Density")
# Plot
axis = None
# axis = [-5., 5., -5., 5.]
Plot.PreparePlot(axis=axis, linewidth=0.5)
Plot.PlotSolution(mesh, EqnSet, solver, "Density", Equidistant=True, PlotExact=False, IncludeMesh2D=True, 
	Regular2D=True, ShowTriangulation=False)
Plot.SaveFigure(FileName=CurrentDir+'vortex', FileType='pdf', CropLevel=2)
Plot.PreparePlot(close_all=False, linewidth=1.5)
Plot.plot_line_probe(mesh, EqnSet, solver, "Density", xy1=[-5.,1.], xy2=[5.,1.], nPoint=101, PlotExact=True, PlotIC=True)
Plot.SaveFigure(FileName=CurrentDir+'line', FileType='pdf', CropLevel=2)
Plot.ShowPlot()

# U = EqnSet.U.Arrays[0]
# code.interact(local=locals())
