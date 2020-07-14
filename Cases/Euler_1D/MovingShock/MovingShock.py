import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import solver.DG as Solver
import physics.euler.euler as Euler
import meshing.common as MeshCommon
import processing.post as Post
import processing.plot as Plot
import general
import os


CurrentDir = os.path.dirname(os.path.abspath(__file__)) + "/"


### Mesh
mesh = MeshCommon.mesh_1D(Uniform=True, nElem=100, xmin=0., xmax=1., Periodic=False)


### Solver parameters
EndTime = 4.e-5
NumTimeSteps = 100
InterpOrder = 1
Params = general.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,NumTimeSteps=NumTimeSteps,
								 InterpBasis="LagrangeSeg",TimeScheme="SSPRK3",InterpolateIC=False,
								 ApplyLimiter="PositivityPreserving")


### Physics
EqnSet = Euler.Euler1D(Params["InterpOrder"], Params["InterpBasis"], mesh)
EqnSet.set_physical_params(GasConstant=287.,SpecificHeatRatio=1.4)
EqnSet.set_conv_num_flux("Roe")
# Parameters
M = 5.
xshock = 0.2
# Initial conditions
EqnSet.set_IC(IC_type="MovingShock", M=M, xshock=xshock)
# Exact solution
EqnSet.set_exact(exact_type="MovingShock", M=M, xshock = xshock)

# Boundary conditions
# for ibfgrp in range(mesh.nBFaceGroup):
# 	BFG = mesh.BFaceGroups[ibfgrp]
# 	if BFG.Name is "Left":
# 		EqnSet.set_BC(BC_type="StateAll", fcn_type="MovingShock", M=M, xshock=xshock)
# 	elif BFG.Name is "Right":
# 		EqnSet.set_BC(BC_type="StateAll", fcn_type="MovingShock", M=M, xshock=xshock)

EqnSet.set_BC(bname="Left", BC_type="StateAll", fcn_type="MovingShock", M=M, xshock=xshock)
EqnSet.set_BC(bname="Right", BC_type="StateAll", fcn_type="MovingShock", M=M, xshock=xshock)

# EqnSet.set_BC(BC_type="StateAll", fcn_type="MovingShock")
# EqnSet.SetBC("Right",Function=EqnSet.FcnMovingShock, BCType=EqnSet.BCType["StateAll"], M=M, xshock=xshock)


### Solve
solver = Solver.DG(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, EqnSet, solver, "Density")
# Plot
# Plot.PreparePlot()
# Plot.PlotSolution(mesh, EqnSet, solver, "Pressure", PlotExact=True, Equidistant=True)
# Plot.SaveFigure(FileName=CurrentDir+'Pressure', FileType='pdf', CropLevel=2)
# Plot.ShowPlot()


# code.interact(local=locals())
