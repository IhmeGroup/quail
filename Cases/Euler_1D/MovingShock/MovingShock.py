import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import Solver
import Euler
import MeshCommon
import Post
import Plot
import General


### Mesh
mesh = MeshCommon.Mesh1D(Uniform=True, nElem=100, xmin=0., xmax=1., Periodic=False)


### Solver parameters
EndTime = 0.0002
nTimeStep = 500
InterpOrder = 1
Params = General.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
								 InterpBasis="SegLagrange",TimeScheme="RK4",InterpolateIC=False,
								 ApplyLimiter="PositivityPreserving")


### Physics
EqnSet = Euler.Euler1D(Params["InterpOrder"], Params["InterpBasis"], mesh, StateRank=3)
EqnSet.SetParams(GasConstant=287.,SpecificHeatRatio=1.4,ConvFlux="Roe")
# Parameters
M = 5.
xshock = 0.2
# Initial conditions
EqnSet.IC.Set(Function=EqnSet.FcnMovingShock, M=M, xshock=xshock)
# Exact solution
EqnSet.ExactSoln.Set(Function=EqnSet.FcnMovingShock, M=M, xshock=xshock)
# Boundary conditions
EqnSet.SetBC("Left",Function=EqnSet.FcnMovingShock, BCType=EqnSet.BCType["FullState"], M=M, xshock=xshock)
EqnSet.SetBC("Right",Function=EqnSet.FcnMovingShock, BCType=EqnSet.BCType["FullState"], M=M, xshock=xshock)


### Solve
solver = Solver.DG_Solver(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
# TotErr,_ = Post.L2_error(mesh, EqnSet, solver.Time, "Density")
# Plot
Plot.PreparePlot()
Plot.PlotSolution(mesh, EqnSet, solver.Time, "Pressure", PlotExact=True, Equidistant=True)
Plot.SaveFigure(FileName='Pressure', FileType='pdf', CropLevel=2)
Plot.ShowPlot()


# code.interact(local=locals())
