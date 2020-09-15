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
import meshing.tools as MeshTools


CurrentDir = os.path.dirname(os.path.abspath(__file__)) + "/"

Periodic = False
num_elems_x = 8
mesh = MeshCommon.mesh_2D(num_elems_x= num_elems_x, num_elems_y = num_elems_x, xmin=-5., xmax=5., 
	ymin=-5., ymax=5.)

### Solver parameters
# SolutionBasis = "LagrangeEqTri"
SolutionBasis = "HierarchicH1Tri"
# SolutionBasis = "LagrangeEqQuad"
if SolutionBasis == "LagrangeEqTri" or SolutionBasis == "HierarchicH1Tri":
	mesh = MeshCommon.split_quadrils_into_tris(mesh)
if Periodic:
	MeshTools.make_periodic_translational(mesh, x1="x1", x2="x2", y1="y1", y2="y2")

dt = 0.025
FinalTime = .5
NumTimeSteps = int(FinalTime/dt + 10.*general.eps)
SolutionOrder = 6

Params = general.set_solver_params(SolutionOrder=SolutionOrder,FinalTime=FinalTime,NumTimeSteps=NumTimeSteps,
								 SolutionBasis=SolutionBasis,TimeStepper="RK4",L2InitialCondition=True,
								 ApplyLimiter=None,WriteInterval=50)


### Physics
x0 = np.array([0., 0.])
physics = Scalar.ConstAdvScalar2D(Params["SolutionOrder"], Params["SolutionBasis"], mesh)
physics.set_physical_params(ConstXVelocity=1.,ConstYVelocity=1.)
physics.set_conv_num_flux("LaxFriedrichs")
# Initial conditions
# physics.IC.Set(Function=physics.FcnGaussian, x0=x0)
physics.set_IC(IC_type="Gaussian", x0=x0)
# Exact solution
# physics.exact_soln.Set(Function=physics.FcnGaussian, x0=x0)
physics.set_exact(exact_type="Gaussian", x0=x0)
# Boundary conditions
# if not Periodic:
# 	MeshTools.check_face_orientations(mesh)
# 	physics.SetBC("x1",Function=physics.FcnGaussian, x0=x0, BCType=physics.BCType["StateAll"])
# 	physics.SetBC("x2",Function=physics.FcnGaussian, x0=x0, BCType=physics.BCType["StateAll"])
# 	physics.SetBC("y1",Function=physics.FcnGaussian, x0=x0, BCType=physics.BCType["StateAll"])
# 	physics.SetBC("y2",Function=physics.FcnGaussian, x0=x0, BCType=physics.BCType["StateAll"])
if not Periodic:
	MeshTools.check_face_orientations(mesh)
	# physics.set_BC(BC_type="StateAll", fcn_type="Gaussian", x0=x0)
	# physics.set_BC(BC_type="StateAll", fcn_type="Gaussian", x0=x0)
	# physics.set_BC(BC_type="StateAll", fcn_type="Gaussian", x0=x0)
	# physics.set_BC(BC_type="StateAll", fcn_type="Gaussian", x0=x0)

	physics.set_BC(bname="x1", BC_type="StateAll", fcn_type="Gaussian", x0=x0)
	physics.set_BC(bname="x2", BC_type="StateAll", fcn_type="Gaussian", x0=x0)
	physics.set_BC(bname="y1", BC_type="StateAll", fcn_type="Gaussian", x0=x0)
	physics.set_BC(bname="y2", BC_type="StateAll", fcn_type="Gaussian", x0=x0)

# raise Exception


### Solve
solver = Solver.DG(Params,physics,mesh)
solver.solve()


### Postprocess
# Error
solver.time = 0.
TotErr,_ = Post.get_error(mesh, physics, solver, "Scalar")
# Plot
axis = None
# # axis = [-5., 5., -5., 5.]
# Plot.prepare_plot(axis=axis, linewidth=0.5)
# Plot.PlotSolution(mesh, physics, solver, "Scalar", Equidistant=True, PlotExact=False, include_mesh=True, 
# 	Regular2D=True, show_triangulation=False)
# Plot.save_figure(file_name=CurrentDir+'Gaussian', file_type='pdf', crop_level=2)
# Plot.plot_line_probe(mesh, physics, solver, "Scalar", xy1=[-5.,-5.], xy2=[5.,5.], nPoint=101, PlotExact=True, PlotIC=True)
# # Post.get_boundary_info(mesh, physics, solver, "y1", "Scalar", integrate=True, 
# # 		vec=[0.,1.], dot_normal_with_vec=True, plot_vs_x=True, plot_vs_y=False)
# Plot.show_plot()

# U = physics.U.Arrays[0]
# code.interact(local=locals())
# 
