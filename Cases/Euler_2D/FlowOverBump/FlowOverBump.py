import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import Solver
import Euler
import MeshCommon
import Post
import Plot
import General
import MeshGmsh


### Mesh
folder = "meshes/"
## Quadrilaterals
subfolder = "Quadrilaterals/"; InterpBasis = "QuadLagrange"
FileName = "bump0_q1.msh"; 
# FileName = "bump0.msh"; 
InterpOrder = [0, 1, 2]; nTimeStep = [500, 1000, 1500]; EndTime = [40., 48., 54.]
# FileName = "bump1_q1.msh"
# FileName = "bump1.msh"
# InterpOrder = [0, 1, 2]; nTimeStep = [500, 1000, 1500]; EndTime = [20., 24., 27.]
# FileName = "bump2_q1.msh"
# FileName = "bump2.msh"
# InterpOrder = [0, 1, 2]; nTimeStep = [500, 1000, 1500]; EndTime = [10., 12., 13.5]
## Triangles
# subfolder = "Triangles/"; InterpBasis = "TriLagrange"
MeshFile = folder + subfolder + FileName
mesh = MeshGmsh.ReadGmshFile(MeshFile)

# Plot.PreparePlot(axis=None, linewidth=0.5)
# Plot.PlotMesh2D(mesh)
# Plot.ShowPlot()
# exit()

### Solver parameters
# InterpOrder = 1
# EndTime = 1.
# nTimeStep = 500
# InterpOrder = [0]; nTimeStep = [500]; EndTime = [40.]
# dt = 0.05
# nTimeStep = int(EndTime/dt + 10.*General.eps)
Params = General.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
								 InterpBasis=InterpBasis,TimeScheme="FE",InterpolateIC=False,
								 TrackOutput=False,WriteTimeHistory=False,OrderSequencing=True)


### Physics
EqnSet = Euler.Euler2D(Params["InterpOrder"], Params["InterpBasis"], mesh, StateRank=4)
EqnSet.SetParams(GasConstant=1.,SpecificHeatRatio=1.4,ConvFlux="Roe")
# Initial conditions
Uinflow = np.array([1.0, 0.5916079783099616, 0.0, 2.675])
EqnSet.IC.Set(Function=EqnSet.FcnUniform, State=Uinflow)
# Boundary conditions
EqnSet.SetBC("inflow", Function=EqnSet.FcnUniform, BCType=EqnSet.BCType["FullState"], State=Uinflow)
EqnSet.SetBC("outflow", BCType=EqnSet.BCType["PressureOutflow"], p=1.)
EqnSet.SetBC("top", BCType=EqnSet.BCType["SlipWall"])
EqnSet.SetBC("bottom", BCType=EqnSet.BCType["SlipWall"])
# Exact solution
EqnSet.ExactSoln.Set(Function=EqnSet.FcnUniform, BCType=EqnSet.BCType["FullState"], State=Uinflow)


### Solve
solver = Solver.DG_Solver(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, EqnSet, solver.Time, "Entropy", NormalizeByVolume=False)
# Plot
axis = None
EqualAR = False
# axis = [-5., 5., -5., 5.]
Plot.PreparePlot(axis=axis, linewidth=0.5)
Plot.PlotSolution(mesh, EqnSet, solver.Time, "Pressure", Equidistant=True, PlotExact=False, IncludeMesh2D=True, 
	ShowTriangulation=False, EqualAR=EqualAR)
Plot.SaveFigure(FileName='Pressure', FileType='pdf', CropLevel=2)
Plot.PlotSolution(mesh, EqnSet, solver.Time, "Entropy", Equidistant=True, PlotExact=False, IncludeMesh2D=True, 
	ShowTriangulation=False, EqualAR=EqualAR)
Plot.SaveFigure(FileName='Entropy', FileType='pdf', CropLevel=2)
Plot.ShowPlot()

# U = EqnSet.U.Arrays[0]
# code.interact(local=locals())