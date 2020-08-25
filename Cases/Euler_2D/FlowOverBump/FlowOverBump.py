import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import solver.DG as Solver
import physics.euler.euler as Euler
import meshing.common as MeshCommon
import processing.post as Post
import processing.plot as Plot
import general
import meshing.gmsh as MeshGmsh
import os


CurrentDir = os.path.dirname(os.path.abspath(__file__)) + "/"

### Mesh
folder = "meshes/"
## Quadrilaterals
subfolder = "Quadrilaterals/"; SolutionBasis = "LagrangeQuad"
#file_name = "bump0_q1.msh"; 
file_name = "bump0.msh"; 
#SolutionOrder = 0; NumTimeSteps = 500; FinalTime = 40.

SolutionOrder = [0, 1, 2]; NumTimeSteps = [500, 1000, 1500]; FinalTime = [40., 48., 54.]
# file_name = "bump1_q1.msh"
# file_name = "bump1.msh"
# SolutionOrder = [0, 1, 2]; NumTimeSteps = [500, 1000, 1500]; FinalTime = [20., 24., 27.]
# # file_name = "bump2_q1.msh"
# file_name = "bump2.msh"
# SolutionOrder = [0, 1, 2]; NumTimeSteps = [500, 1000, 1500]; FinalTime = [10., 12., 13.5]
# Triangles
#subfolder = "Triangles/"; SolutionBasis = "LagrangeEqTri"
#file_name = "bump0_tri.msh"
MeshFile = CurrentDir + folder + subfolder + file_name
mesh = MeshGmsh.import_gmsh_mesh(MeshFile)

# Plot.prepare_plot(axis=None, linewidth=0.5)
# Plot.plot_mesh(mesh)
# Plot.show_plot()
# exit()

### Solver parameters
# SolutionOrder = 1
# FinalTime = 1.
# NumTimeSteps = 500
# SolutionOrder = [0]; NumTimeSteps = [500]; FinalTime = [40.]
# dt = 0.05
# NumTimeSteps = int(EndTime/dt + 10.*general.eps)
Params = general.SetSolverParams(SolutionOrder=SolutionOrder,FinalTime=FinalTime,NumTimeSteps=NumTimeSteps,
								 SolutionBasis=SolutionBasis,TimeStepper="FE",L2InitialCondition=True,
								 TrackOutput=False,WriteTimeHistory=False,OrderSequencing=True)


### Physics
physics = Euler.Euler2D(Params["SolutionOrder"], Params["SolutionBasis"], mesh)
physics.set_physical_params(GasConstant=1.,SpecificHeatRatio=1.4)
physics.set_conv_num_flux("Roe")
# Initial conditions
Uinflow = np.array([1.0, 0.5916079783099616, 0.0, 2.675])
# physics.IC.Set(Function=physics.FcnUniform, State=Uinflow)
physics.set_IC(IC_type="Uniform", state=Uinflow)
physics.set_exact(exact_type="Uniform", state=Uinflow)
# Boundary conditions
# physics.SetBC("inflow", Function=physics.FcnUniform, BCType=physics.BCType["StateAll"], State=Uinflow)
# physics.SetBC("outflow", BCType=physics.BCType["PressureOutflow"], p=1.)
# physics.SetBC("top", BCType=physics.BCType["SlipWall"])
# physics.SetBC("bottom", BCType=physics.BCType["SlipWall"])
# for ibfgrp in range(mesh.num_boundary_groups):
# 	BFG = mesh.boundary_groups[ibfgrp]
# 	if BFG.Name == "inflow":
# 		physics.set_BC(BC_type="StateAll", fcn_type="Uniform", state=Uinflow)
# 	elif BFG.Name == "outflow":
# 		physics.set_BC(BC_type="PressureOutlet", p=1.)
# 	elif BFG.Name == "top":
# 		physics.set_BC(BC_type="SlipWall")
# 	elif BFG.Name == "bottom":
# 		physics.set_BC(BC_type="SlipWall")
# 	else:
# 		raise Exception

physics.set_BC(bname="inflow", BC_type="StateAll", fcn_type="Uniform", state=Uinflow)
physics.set_BC(bname="outflow", BC_type="PressureOutlet", p=1.)
physics.set_BC(bname="top", BC_type="SlipWall")
physics.set_BC(bname="bottom", BC_type="SlipWall")
# Exact solution
# physics.ExactSoln.Set(Function=physics.FcnUniform, BCType=physics.BCType["StateAll"], State=Uinflow)


### Solve
solver = Solver.DG(Params,physics,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.get_error(mesh, physics, solver, "Entropy", NormalizeByVolume=False)
# Plot
axis = None
equal_AR = False
# axis = [-5., 5., -5., 5.]
# Plot.prepare_plot(axis=axis, linewidth=0.5)
# Plot.PlotSolution(mesh, physics, solver, "Pressure", Equidistant=True, PlotExact=False, include_mesh=True, 
# 	show_triangulation=False, equal_AR=equal_AR, show_elem_IDs=True)
# Plot.save_figure(file_name='Pressure', FileType='pdf', CropLevel=2)
# Plot.PlotSolution(mesh, physics, solver, "Entropy", Equidistant=True, PlotExact=False, include_mesh=True, 
# 	show_triangulation=False, equal_AR=equal_AR)
# Plot.save_figure(file_name=CurrentDir+'Entropy', FileType='pdf', CropLevel=2)
# Post.get_boundary_info(mesh, physics, solver, "bottom", "Pressure", integrate=True, 
# 		vec=[1.,0.], dot_normal_with_vec=True, plot_vs_x=True, plot_vs_y=False, Label="F_x")
# Plot.show_plot()

# U = physics.U.Arrays[0]
# code.interact(local=locals())
