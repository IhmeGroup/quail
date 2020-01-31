import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import Solver
import Scalar
import MeshCommon
import Post
import Plot
import General
import Limiter
### Mesh
Periodic = False
# Uniform mesh
mesh = MeshCommon.Mesh1D(Uniform=True, nElem=50, xmin=0., xmax=1., Periodic=Periodic)
# Non-uniform mesh
# nElem = 25
# Coords = np.cos(np.linspace(np.pi,0.,nElem+1))
# Coords = MeshCommon.RefineUniform1D(Coords)
# # Coords = MeshCommon.RefineUniform1D(Coords)
# mesh = MeshCommon.Mesh1D(Coords=Coords, Periodic=Periodic)


### Solver parameters
dt = 0.001
mu = 1.
EndTime = 0.3
#nTimeStep = np.amax([1,int(EndTime/((mesh.Coords[1,0] - mesh.Coords[0,0])*0.1))])
nTimeStep = int(EndTime/dt)
InterpOrder = 2
Params = General.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
								 InterpBasis="SegLagrange",TimeScheme="SSPRK3",
								 ApplyLimiter="ScalarPositivityPreserving")

### Physics
ConstVelocity = 1.
EqnSet = Scalar.Scalar(Params["InterpOrder"], Params["InterpBasis"], mesh, StateRank=1)
EqnSet.SetParams(AdvectionOperator="Burgers")
EqnSet.SetParams(ConstVelocity=ConstVelocity, ConvFlux="LaxFriedrichs")

# Initial conditions
EqnSet.IC.Set(Function=EqnSet.FcnScalarShock, uL = 1., uR = 0.,  xshock = 0.3)
# Exact solution
EqnSet.ExactSoln.Set(Function=EqnSet.FcnScalarShock, uL = 1., uR = 0., xshock = 0.3)
# Boundary conditions
if ConstVelocity >= 0.:
	Inflow = "Left"; Outflow = "Right"
else:
	Inflow = "Right"; Outflow = "Left"
if not Periodic:
	for ibfgrp in range(mesh.nBFaceGroup):
		BC = EqnSet.BCs[ibfgrp]
		## Left
		if BC.Name is Inflow:
			BC.Set(Function=EqnSet.FcnScalarShock, BCType=EqnSet.BCType["FullState"], uL = 1., uR = 0., xshock = 0.3)
		elif BC.Name is Outflow:
			BC.Set(BCType=EqnSet.BCType["Extrapolation"])
			# BC.Set(Function=EqnSet.FcnSine, BCType=EqnSet.BCType["FullState"], omega = 2*np.pi)
		else:
			raise Exception("BC error")


### Solve
solver = Solver.DG_Solver(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, EqnSet, solver.Time, "Scalar")
# Plot
Plot.PreparePlot()
Plot.PlotSolution(mesh, EqnSet, solver.Time, "Scalar", PlotExact = True, Label="u")
Plot.ShowPlot()


# code.interact(local=locals())
