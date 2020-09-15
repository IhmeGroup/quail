import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import solver.DG as Solver
import physics.scalar.scalar as Scalar
import meshing.common as MeshCommon
import processing.post as Post
import processing.plot as Plot
import general
import meshing.gmsh as MeshGmsh
import os


CurrentDir = os.path.dirname(os.path.abspath(__file__)) + "/"


### Mesh
folder = "meshes/"
# Quadrilaterals
# subfolder = "Quadrilaterals/"; SolutionBasis = "LagrangeEqQuad"
# # # Structured
# subsubfolder = "Structured/"
# file_name = "box_5x5.msh"
# file_name = "box_10x10.msh"
# file_name = "box_20x20.msh"
# file_name = "box_40x40.msh"
# file_name = "box_80x80.msh"
# Unstructured
# subsubfolder = "Unstructured/"
# file_name = "box_25_elem.msh"
# file_name = "box_100_elem.msh"
# file_name = "box_400_elem.msh"
# file_name = "box_1600_elem.msh"
## Triangles
subfolder = "Triangles/"; SolutionBasis = "HierarchicH1Tri"
# Structured
subsubfolder = "Structured/"
file_name = "box_5x5.msh"
# file_name = "box_10x10.msh"
# file_name = "box_20x20.msh"
# file_name = "box_40x40.msh"
MeshFile = os.path.dirname(os.path.abspath(__file__)) + "/" + folder + subfolder + subsubfolder + file_name
mesh = MeshGmsh.import_gmsh_mesh(MeshFile)

### Solver parameters
dt = 0.05
FinalTime = 2.0
NumTimeSteps = int(FinalTime/dt + 10.*general.eps)
SolutionOrder = 2
Params = general.set_solver_params(SolutionOrder=SolutionOrder,FinalTime=FinalTime,NumTimeSteps=NumTimeSteps,
								 SolutionBasis=SolutionBasis,TimeStepper="RK4",L2InitialCondition=True,
								 ElementQuadrature="Dunavant",ApplyLimiter=None)


### Physics
x0 = np.array([-0.5,-0.2])
physics = Scalar.ConstAdvScalar2D(Params["SolutionOrder"], Params["SolutionBasis"], mesh)
# physics.set_physical_params(ConstXVelocity=1.,ConstYVelocity=0.5)
physics.set_physical_params(ConstXVelocity=1.,ConstYVelocity=0.5)
physics.set_conv_num_flux("LaxFriedrichs")
# Initial conditions
physics.set_IC(IC_type="Gaussian", x0=x0)
physics.set_exact(exact_type="Gaussian",x0=x0)

# Boundary conditions
physics.set_BC(bname="wall", BC_type="StateAll", fcn_type="Gaussian", x0=x0)
# physics.SetBC("wall",Function=physics.FcnGaussian, x0=x0, BCType=physics.BCType["StateAll"])
# raise Exception


### Solve
solver = Solver.DG(Params,physics,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.get_error(mesh, physics, solver, "Scalar")
# Plot
axis = None
# axis = [-5., 5., -5., 5.]
# Plot.prepare_plot(axis=axis, linewidth=0.5)
# Plot.PlotSolution(mesh, physics, solver, "Scalar", Equidistant=True, PlotExact=False, include_mesh=True, 
# 	Regular2D=True, show_triangulation=False)
# Plot.save_figure(file_name=CurrentDir+'Gaussian', file_type='pdf', crop_level=2)
# Plot.show_plot()

# U = physics.U.Arrays[0]
# code.interact(local=locals())
