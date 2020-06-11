import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import solver.ADERDG as Solver
import physics.euler.euler as Euler
import meshing.common as MeshCommon
import processing.post as Post
import processing.plot as Plot
import general



### Mesh
Periodic = True
mesh = MeshCommon.mesh_1D(Uniform=True, nElem=20, xmin=0., xmax=1., Periodic=Periodic)


nu = -1000.

### Solver parameters
EndTime = 0.5
nTimeStep = np.amax([1,int(EndTime/((mesh.Coords[1,0] - mesh.Coords[0,0])*0.075))])
# nTimeStep = 100
InterpOrder = 2
Params = general.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
								 InterpBasis="LagrangeEqSeg",SourceTreatment="Implicit")


### Physics
EqnSet = Euler.Euler1D(Params["InterpOrder"], Params["InterpBasis"], mesh)
EqnSet.SetParams(SpecificHeatRatio=1.4,ConvFlux="Roe")

# Initial conditions
EqnSet.set_IC(IC_type="DensityWave", p=1.0)
EqnSet.set_source(source_type="StiffFriction",nu=nu)

### Solve
solver = Solver.ADERDG_Solver(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
# TotErr,_ = Post.L2_error(mesh, EqnSet, solver, "Density")
# Plot
Plot.PreparePlot()
Plot.PlotSolution(mesh, EqnSet, solver, "Density", PlotIC=True, PlotExact=False, Equidistant=True)
Plot.PlotSolution(mesh, EqnSet, solver, "XMomentum", PlotIC=True, PlotExact=False, Equidistant=True)
Plot.PlotSolution(mesh, EqnSet, solver, "Energy", PlotIC=True, PlotExact=False, Equidistant=True)

Plot.ShowPlot()
