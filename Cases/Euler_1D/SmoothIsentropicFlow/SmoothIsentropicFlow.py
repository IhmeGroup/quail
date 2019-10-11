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
mesh = MeshCommon.Mesh1D(Uniform=True, nElem=25, xmin=-1., xmax=1., Periodic=Periodic)


### Solver parameters
EndTime = 0.1
nTimeStep = 100
InterpOrder = 2
Params = General.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
								 InterpBasis="SegLagrange",TimeScheme="RK4")


### Physics
EqnSet = Euler.Euler(Params["InterpOrder"], Params["InterpBasis"], mesh, StateRank=3)
EqnSet.SetParams(GasConstant=1.,SpecificHeatRatio=3.,ConvFlux="Roe")
# Initial conditions
EqnSet.IC.Set(Function=EqnSet.FcnSmoothIsentropicFlow, a=0.9)
# Exact solution
EqnSet.ExactSoln.Set(Function=EqnSet.FcnSmoothIsentropicFlow, a=0.9)
# Boundary conditions
if not Periodic:
	for ibfgrp in range(mesh.nBFaceGroup):
		BC = EqnSet.BCs[ibfgrp]
		## Left
		if BC.Title is "Left":
			BC.Set(Function=EqnSet.FcnSmoothIsentropicFlow, BCType=EqnSet.BCType["FullState"], a=0.9)
		elif BC.Title is "Right":
			BC.Set(Function=EqnSet.FcnSmoothIsentropicFlow, BCType=EqnSet.BCType["FullState"], a=0.9)
			# BC.Set(BCType=EqnSet.BCType["Extrapolation"])
		else:
			raise Exception("BC error")


### Solve
solver = Solver.DG_Solver(Params,EqnSet,mesh)
solver.ApplyTimeScheme()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, EqnSet, EndTime, "Density")
# Plot
Plot.PreparePlot()
Plot.Plot1D(mesh, EqnSet, EndTime, "XMomentum")


# code.interact(local=locals())
