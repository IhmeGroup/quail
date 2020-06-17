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
subfolder = "Quadrilaterals/"; InterpBasis = "LagrangeEqQuad"
#FileName = "bump0_q1.msh"; 
FileName = "bump0.msh"; 
#InterpOrder = 0; nTimeStep = 500; EndTime = 40.

InterpOrder = [0, 1, 2]; nTimeStep = [500, 1000, 1500]; EndTime = [40., 48., 54.]
# FileName = "bump1_q1.msh"
# FileName = "bump1.msh"
# InterpOrder = [0, 1, 2]; nTimeStep = [500, 1000, 1500]; EndTime = [20., 24., 27.]
# # FileName = "bump2_q1.msh"
# FileName = "bump2.msh"
# InterpOrder = [0, 1, 2]; nTimeStep = [500, 1000, 1500]; EndTime = [10., 12., 13.5]
# Triangles
#subfolder = "Triangles/"; InterpBasis = "LagrangeEqTri"
#FileName = "bump0_tri.msh"
MeshFile = CurrentDir + folder + subfolder + FileName
mesh = MeshGmsh.ReadGmshFile(MeshFile)

# Plot.PreparePlot(axis=None, linewidth=0.5)
# Plot.plot_mesh(mesh)
# Plot.ShowPlot()
# exit()

### Solver parameters
# InterpOrder = 1
# EndTime = 1.
# nTimeStep = 500
# InterpOrder = [0]; nTimeStep = [500]; EndTime = [40.]
# dt = 0.05
# nTimeStep = int(EndTime/dt + 10.*general.eps)
Params = general.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
								 InterpBasis=InterpBasis,TimeScheme="FE",InterpolateIC=False,
								 TrackOutput=False,WriteTimeHistory=False,OrderSequencing=True)


### Physics
EqnSet = Euler.Euler2D(Params["InterpOrder"], Params["InterpBasis"], mesh)
EqnSet.set_physical_params(GasConstant=1.,SpecificHeatRatio=1.4)
EqnSet.set_conv_num_flux("Roe")
# Initial conditions
Uinflow = np.array([1.0, 0.5916079783099616, 0.0, 2.675])
# EqnSet.IC.Set(Function=EqnSet.FcnUniform, State=Uinflow)
EqnSet.set_IC(IC_type="Uniform", state=Uinflow)
EqnSet.set_exact(exact_type="Uniform", state=Uinflow)
# Boundary conditions
# EqnSet.SetBC("inflow", Function=EqnSet.FcnUniform, BCType=EqnSet.BCType["StateAll"], State=Uinflow)
# EqnSet.SetBC("outflow", BCType=EqnSet.BCType["PressureOutflow"], p=1.)
# EqnSet.SetBC("top", BCType=EqnSet.BCType["SlipWall"])
# EqnSet.SetBC("bottom", BCType=EqnSet.BCType["SlipWall"])
for ibfgrp in range(mesh.nBFaceGroup):
	BFG = mesh.BFaceGroups[ibfgrp]
	if BFG.Name == "inflow":
		EqnSet.set_BC(BC_type="StateAll", fcn_type="Uniform", state=Uinflow)
	elif BFG.Name == "outflow":
		EqnSet.set_BC(BC_type="PressureOutlet", p=1.)
	elif BFG.Name == "top":
		EqnSet.set_BC(BC_type="SlipWall")
	elif BFG.Name == "bottom":
		EqnSet.set_BC(BC_type="SlipWall")
	else:
		raise Exception
# Exact solution
# EqnSet.ExactSoln.Set(Function=EqnSet.FcnUniform, BCType=EqnSet.BCType["StateAll"], State=Uinflow)


### Solve
solver = Solver.DG(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, EqnSet, solver, "Entropy", NormalizeByVolume=False)
# Plot
axis = None
EqualAR = False
# axis = [-5., 5., -5., 5.]
Plot.PreparePlot(axis=axis, linewidth=0.5)
Plot.PlotSolution(mesh, EqnSet, solver, "Pressure", Equidistant=True, PlotExact=False, include_mesh=True, 
	ShowTriangulation=False, EqualAR=EqualAR, show_elem_IDs=True)
Plot.SaveFigure(FileName='Pressure', FileType='pdf', CropLevel=2)
Plot.PlotSolution(mesh, EqnSet, solver, "Entropy", Equidistant=True, PlotExact=False, include_mesh=True, 
	ShowTriangulation=False, EqualAR=EqualAR)
Plot.SaveFigure(FileName=CurrentDir+'Entropy', FileType='pdf', CropLevel=2)
Post.get_boundary_info(mesh, EqnSet, solver, "bottom", "Pressure", integrate=True, 
		vec=[1.,0.], dot_normal_with_vec=True, plot_vs_x=True, plot_vs_y=False, Label="F_x")
Plot.ShowPlot()

# U = EqnSet.U.Arrays[0]
# code.interact(local=locals())
