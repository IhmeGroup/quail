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
mesh = MeshCommon.mesh_1D(Uniform=True, num_elems=20, xmin=0., xmax=1., Periodic=Periodic)


nu = -1000.

### Solver parameters
FinalTime = 0.5
NumTimeSteps = np.amax([1,int(FinalTime/((mesh.node_coords[-1,0] - mesh.node_coords[-2,0])*0.075))])
# NumTimeSteps = 100
SolutionOrder = 2
Params = general.set_solver_params(SolutionOrder=SolutionOrder,FinalTime=FinalTime,NumTimeSteps=NumTimeSteps,
								 SolutionBasis="LagrangeSeg",SourceTreatmentADER="Implicit")


### Physics
physics = Euler.Euler1D(Params["SolutionOrder"], Params["SolutionBasis"], mesh)
physics.set_physical_params(SpecificHeatRatio=1.4)
physics.set_conv_num_flux("Roe")

# Initial conditions
physics.set_IC(IC_type="DensityWave", p=1.0)
physics.set_source(source_type="StiffFriction",nu=nu)

### Solve
solver = Solver.ADERDG(Params,physics,mesh)
solver.solve()


### Postprocess
# Error
# TotErr,_ = Post.get_error(mesh, physics, solver, "Density")
# Plot
Plot.prepare_plot()
Plot.PlotSolution(mesh, physics, solver, "Density", PlotIC=True, PlotExact=False, Equidistant=True)
Plot.PlotSolution(mesh, physics, solver, "XMomentum", PlotIC=True, PlotExact=False, Equidistant=True)
Plot.PlotSolution(mesh, physics, solver, "Energy", PlotIC=True, PlotExact=False, Equidistant=True)

Plot.show_plot()
