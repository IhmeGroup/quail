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
Periodic = False
mesh = MeshCommon.mesh_1D(Uniform=True, nElem=25, xmin=-1., xmax=1., Periodic=Periodic)


### Solver parameters
EndTime = 0.1
nTimeStep = 100
InterpOrder = 2
Params = General.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
								 InterpBasis="LagrangeEqSeg",TimeScheme="ADER",InterpolateIC=True)


### Physics
EqnSet = Euler.Euler1D(Params["InterpOrder"], Params["InterpBasis"], mesh, StateRank=3)
EqnSet.SetParams(GasConstant=1.,SpecificHeatRatio=3.,ConvFlux="LaxFriedrichs")
# Initial conditions
EqnSet.IC.Set(Function=EqnSet.FcnSmoothIsentropicFlow, a=0.9)
# Exact solution
EqnSet.ExactSoln.Set(Function=EqnSet.FcnSmoothIsentropicFlow, a=0.9)
# Boundary conditions
if not Periodic:
	EqnSet.SetBC("Left",Function=EqnSet.FcnSmoothIsentropicFlow, BCType=EqnSet.BCType["FullState"], a=0.9)
	EqnSet.SetBC("Right",Function=EqnSet.FcnSmoothIsentropicFlow, BCType=EqnSet.BCType["FullState"], a=0.9)


### Solve
solver = Solver.ADERDG_Solver(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, EqnSet, solver, "Density")
# Plot
Plot.PreparePlot()
Plot.PlotSolution(mesh, EqnSet, solver, "Energy", PlotExact=True, Equidistant=True)
Plot.ShowPlot()


# code.interact(local=locals())
