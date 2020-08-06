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
mesh = MeshCommon.mesh_2D(xcoords=None, ycoords=None, num_elems_x= num_elems_x, num_elems_y = num_elems_x, Uniform=True, xmin=-5., xmax=5., 
	ymin=-5., ymax=5., Periodic=Periodic)

### Solver parameters
# InterpBasis = "LagrangeEqTri"
InterpBasis = "HierarchicH1Tri"
# InterpBasis = "LagrangeEqQuad"
if InterpBasis == "LagrangeEqTri" or InterpBasis == "HierarchicH1Tri":
	mesh = MeshCommon.split_quadrils_into_tris(mesh)
if Periodic:
	MeshTools.MakePeriodicTranslational(mesh, x1="x1", x2="x2", y1="y1", y2="y2")

dt = 0.025
EndTime = .5
NumTimeSteps = int(EndTime/dt + 10.*general.eps)
InterpOrder = 6
Params = general.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,NumTimeSteps=NumTimeSteps,
								 InterpBasis=InterpBasis,TimeScheme="RK4",InterpolateIC=False,
								 ApplyLimiter=None,WriteInterval=50)


### Physics
x0 = np.array([0., 0.])
physics = Scalar.ConstAdvScalar2D(Params["InterpOrder"], Params["InterpBasis"], mesh)
physics.set_physical_params(ConstXVelocity=1.,ConstYVelocity=1.)
physics.set_conv_num_flux("LaxFriedrichs")
# Initial conditions
# physics.IC.Set(Function=physics.FcnGaussian, x0=x0)
physics.set_IC(IC_type="Gaussian", x0=x0)
# Exact solution
# physics.ExactSoln.Set(Function=physics.FcnGaussian, x0=x0)
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
solver.Time = 0.
TotErr,_ = Post.L2_error(mesh, physics, solver, "Scalar")
# Plot
axis = None
# axis = [-5., 5., -5., 5.]
Plot.PreparePlot(axis=axis, linewidth=0.5)
Plot.PlotSolution(mesh, physics, solver, "Scalar", Equidistant=True, PlotExact=False, include_mesh=True, 
	Regular2D=True, show_triangulation=False)
Plot.SaveFigure(FileName=CurrentDir+'Gaussian', FileType='pdf', CropLevel=2)
Plot.plot_line_probe(mesh, physics, solver, "Scalar", xy1=[-5.,-5.], xy2=[5.,5.], nPoint=101, PlotExact=True, PlotIC=True)
# Post.get_boundary_info(mesh, physics, solver, "y1", "Scalar", integrate=True, 
# 		vec=[0.,1.], dot_normal_with_vec=True, plot_vs_x=True, plot_vs_y=False)
Plot.ShowPlot()

# U = physics.U.Arrays[0]
# code.interact(local=locals())
# 
