import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import Solver
import physics.scalar.scalar as Scalar
import meshing.common as MeshCommon
import Post
import Plot
import General
import Limiter
### Mesh
Periodic = False
# Uniform mesh
mesh = MeshCommon.mesh_1D(Uniform=True, nElem=50, xmin=0., xmax=1., Periodic=Periodic)
# Non-uniform mesh
# nElem = 25
# Coords = np.cos(np.linspace(np.pi,0.,nElem+1))
# Coords = MeshCommon.refine_uniform_1D(Coords)
# # Coords = MeshCommon.refine_uniform_1D(Coords)
# mesh = MeshCommon.mesh_1D(Coords=Coords, Periodic=Periodic)


### Solver parameters
#dt = 0.001
mu = 1.
EndTime = 0.3
nTimeStep = np.amax([1,int(EndTime/((mesh.Coords[1,0] - mesh.Coords[0,0])*0.1))])
#nTimeStep = int(EndTime/dt)
InterpOrder = 2
Params = General.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
								 InterpBasis="LagrangeEqSeg",TimeScheme="ADER")
								 #ApplyLimiter="ScalarPositivityPreserving")

### Physics
Velocity = 1.0
EqnSet = Scalar.Burgers(Params["InterpOrder"], Params["InterpBasis"], mesh, StateRank=1)
EqnSet.SetParams(ConstVelocity=Velocity)
#EqnSet.SetParams(AdvectionOperator="Burgers")
EqnSet.SetParams(ConvFlux="LaxFriedrichs")


# ----------------------------------------------------------------------------------------------------
# This case is designed to test the time integration schemes ability to handle a stiff source term.
# In this function, 'SetSource', we manipulate the stiffness using a parameter that as it approaches
# zero, increases the amount of stiffness in the solution.
#
# For Example: If stiffness is set to 0.1, then the equation is not very stiff. But, if it is set to 
# something lower, (i.e. 0.001) then you will observe a stable solution, but the location of the shock
# will not be correct. It will have propogated at some other speed. (Note: RK4 cannot run stiff case)
# -----------------------------------------------------------------------------------------------------
EqnSet.SetSource(Function=EqnSet.FcnStiffSource,beta=0.5, stiffness = 1.)
# Initial conditions
EqnSet.IC.Set(Function=EqnSet.FcnScalarShock, uL = 1., uR = 0.,  xshock = 0.3)

# Exact solution
EqnSet.ExactSoln.Set(Function=EqnSet.FcnScalarShock, uL = 1., uR = 0.,  xshock = 0.3)
# Boundary conditions
if Velocity >= 0.:
	Inflow = "Left"; Outflow = "Right"
else:
	Inflow = "Right"; Outflow = "Left"
if not Periodic:
	for ibfgrp in range(mesh.nBFaceGroup):
		BC = EqnSet.BCs[ibfgrp]
		## Left
		if BC.Name is Inflow:
			BC.Set(Function=EqnSet.FcnScalarShock, BCType=EqnSet.BCType["FullState"], uL = 1., uR = 0., xshock = 0.3)
			#BC.Set(Function=EqnSet.FcnUniform, BCType=EqnSet.BCType["FullState"], State = [1.])
		elif BC.Name is Outflow:
			BC.Set(BCType=EqnSet.BCType["Extrapolation"])
			# BC.Set(Function=EqnSet.FcnSine, BCType=EqnSet.BCType["FullState"], omega = 2*np.pi)
		else:
			raise Exception("BC error")


### Solve
solver = Solver.ADERDG_Solver(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
#TotErr,_ = Post.L2_error(mesh, EqnSet, solver.Time, "Scalar")
# Plot
Plot.PreparePlot()
Plot.PlotSolution(mesh, EqnSet, solver.Time, "Scalar", PlotExact = True, PlotIC = True, Label="Q_h")
Plot.ShowPlot()


# code.interact(local=locals())
