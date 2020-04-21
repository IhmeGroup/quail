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
import MeshTools


CurrentDir = os.path.dirname(os.path.abspath(__file__)) + "/"

Periodic = True
nElem_x = 5
mesh = MeshCommon.mesh_2D(xcoords=None, ycoords=None, nElem_x=nElem_x, nElem_y = nElem_x, Uniform=True, xmin=-5., xmax=5., 
	ymin=-5., ymax=5., Periodic=Periodic)
if Periodic:
	MeshTools.MakePeriodicTranslational(mesh, x1="x1", x2="x2", y1="y1", y2="y2")

mesh = MeshCommon.split_quadrils_into_tris(mesh)

### Solver parameters
InterpBasis = "LagrangeTri"
dt = 0.05
EndTime = 10.0
nTimeStep = int(EndTime/dt + 10.*General.eps)
InterpOrder = 2
Params = General.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
								 InterpBasis=InterpBasis,TimeScheme="RK4",InterpolateIC=False,
								 ApplyLimiter=None)


### Physics
x0 = np.array([0., 0.])
EqnSet = Scalar.ConstAdvScalar2D(Params["InterpOrder"], Params["InterpBasis"], mesh)
EqnSet.SetParams(ConstXVelocity=1.,ConstYVelocity=1.,ConvFlux="LaxFriedrichs")
# Initial conditions
EqnSet.IC.Set(Function=EqnSet.FcnGaussian, x0=x0)
# Exact solution
EqnSet.ExactSoln.Set(Function=EqnSet.FcnGaussian, x0=x0)
# Boundary conditions
if not Periodic:
	MeshTools.check_face_orientations(mesh)
	EqnSet.SetBC("x1",Function=EqnSet.FcnGaussian, x0=x0, BCType=EqnSet.BCType["FullState"])
	EqnSet.SetBC("x2",Function=EqnSet.FcnGaussian, x0=x0, BCType=EqnSet.BCType["FullState"])
	EqnSet.SetBC("y1",Function=EqnSet.FcnGaussian, x0=x0, BCType=EqnSet.BCType["FullState"])
	EqnSet.SetBC("y2",Function=EqnSet.FcnGaussian, x0=x0, BCType=EqnSet.BCType["FullState"])
# raise Exception


### Solve
solver = Solver.DG_Solver(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, EqnSet, 0., "Scalar")
# Plot
axis = None
# axis = [-5., 5., -5., 5.]
Plot.PreparePlot(axis=axis, linewidth=0.5)
Plot.PlotSolution(mesh, EqnSet, solver.Time, "Scalar", Equidistant=True, PlotExact=False, IncludeMesh2D=True, 
	Regular2D=True, ShowTriangulation=False)
Plot.SaveFigure(FileName=CurrentDir+'Gaussian', FileType='pdf', CropLevel=2)
Plot.ShowPlot()

# U = EqnSet.U.Arrays[0]
# code.interact(local=locals())
