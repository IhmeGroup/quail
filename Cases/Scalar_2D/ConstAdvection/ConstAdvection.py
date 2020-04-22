import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import Solver
import Scalar
import MeshCommon
import Post
import Plot
import General
import MeshGmsh
import os


CurrentDir = os.path.dirname(os.path.abspath(__file__)) + "/"


### Mesh
folder = "meshes/"
# Quadrilaterals
# subfolder = "Quadrilaterals/"; InterpBasis = "LagrangeQuad"
# # # Structured
# subsubfolder = "Structured/"
# FileName = "box_5x5.msh"
# FileName = "box_10x10.msh"
# FileName = "box_20x20.msh"
# FileName = "box_40x40.msh"
# FileName = "box_80x80.msh"
# Unstructured
# subsubfolder = "Unstructured/"
# FileName = "box_25_elem.msh"
# FileName = "box_100_elem.msh"
# FileName = "box_400_elem.msh"
# FileName = "box_1600_elem.msh"
## Triangles
subfolder = "Triangles/"; InterpBasis = "LagrangeTri"
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
EndTime = 2.0
nTimeStep = int(EndTime/dt + 10.*General.eps)
InterpOrder = 2
Params = General.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
								 InterpBasis=InterpBasis,TimeScheme="RK4",InterpolateIC=False,
								 ApplyLimiter=None)


### Physics
x0 = np.array([-0.5,-0.2])
EqnSet = Scalar.ConstAdvScalar2D(Params["InterpOrder"], Params["InterpBasis"], mesh)
EqnSet.SetParams(ConstXVelocity=1.,ConstYVelocity=0.5,ConvFlux="LaxFriedrichs")
# Initial conditions
EqnSet.IC.Set(Function=EqnSet.FcnGaussian, x0=x0)
# Exact solution
EqnSet.ExactSoln.Set(Function=EqnSet.FcnGaussian, x0=x0)
# Boundary conditions
EqnSet.SetBC("wall",Function=EqnSet.FcnGaussian, x0=x0, BCType=EqnSet.BCType["FullState"])
# raise Exception


### Solve
solver = Solver.DG_Solver(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, EqnSet, solver, "Scalar")
# Plot
axis = None
# axis = [-5., 5., -5., 5.]
Plot.PreparePlot(axis=axis, linewidth=0.5)
Plot.PlotSolution(mesh, EqnSet, solver, "Scalar", Equidistant=True, PlotExact=False, IncludeMesh2D=True, 
	Regular2D=True, ShowTriangulation=False)
Plot.SaveFigure(FileName=CurrentDir+'Gaussian', FileType='pdf', CropLevel=2)
Plot.ShowPlot()

# U = EqnSet.U.Arrays[0]
# code.interact(local=locals())
