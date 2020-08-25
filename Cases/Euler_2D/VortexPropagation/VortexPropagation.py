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
import meshing.tools as MeshTools
import copy


def RandomizeNodes(mesh):
    OldNode2NewNode = np.arange(mesh.num_nodes)
    np.random.shuffle(OldNode2NewNode)
    NewNodeOrder = np.zeros(mesh.num_nodes, dtype=int) - 1
    NewNodeOrder[OldNode2NewNode] = np.arange(mesh.num_nodes)

    if np.min(NewNodeOrder) == -1:
        raise ValueError

    MeshTools.remap_nodes(mesh, OldNode2NewNode, NewNodeOrder)

    # for i in range(mesh.num_nodes):
    #   n = OldNode2NewNode[i]
    #     NewNodeOrder[n] = i
    #         # NewNodeOrder[i] = the node number (pre-reordering) of the ith node (post-reordering)

    print("Randomized nodes")


def RandomizeElements(mesh):
    old_to_new_elem = np.arange(mesh.num_elems)
    np.random.shuffle(old_to_new_elem)
    new_elem_order = np.zeros(mesh.num_elems, dtype=int) - 1
    new_elem_order[old_to_new_elem] = np.arange(mesh.num_elems)

    if np.min(new_elem_order) == -1:
        raise ValueError

    mesh.elem_to_node_ids = mesh.elem_to_node_ids[new_elem_order]
    elements_old = copy.deepcopy(mesh.elements)
    for e in range(mesh.num_elems):
    	mesh.elements[e] = elements_old[new_elem_order[e]]

    for iface in mesh.interior_faces:
    	iface.elemL_id = old_to_new_elem[iface.elemL_id]
    	iface.elemR_id = old_to_new_elem[iface.elemR_id]

    for boundary_group in mesh.boundary_groups.values():
    	for bface in boundary_group.boundary_faces:
    		bface.elem_id = old_to_new_elem[bface.elem_id]


    # Assign new node IDs
    # mesh.node_coords = mesh.node_coords[new_to_old_node_map]

    # # New elem_to_node_ids
    # num_elems = mesh.num_elems
    # for elem_id in range(num_elems):
    #     mesh.elem_to_node_ids[elem_id,:] = old_to_new_node_map[
    #             mesh.elem_to_node_ids[elem_id, :]]

    print("Randomized elements")


CurrentDir = os.path.dirname(os.path.abspath(__file__)) + "/"


### Mesh
folder = "meshes/"
# Quadrilaterals
#subfolder = "Quadrilaterals/"; SolutionBasis = "LagrangeEqQuad"
# # # Structured
#subsubfolder = "Structured/"
#file_name = "box_5x5.msh"
# file_name = "box_10x10.msh"
# file_name = "box_20x20.msh"
# file_name = "box_40x40.msh"
# file_name = "box_80x80.msh"
# Unstructured
#subsubfolder = "Unstructured/"
#file_name = "box_25_elem.msh"
# file_name = "box_100_elem.msh"
# file_name = "box_400_elem.msh"
# file_name = "box_1600_elem.msh"
## Triangles
subfolder = "Triangles/"; SolutionBasis = "LagrangeTri"
# Structured
# subsubfolder = "Structured/"
# file_name = "box_5x5_v4.msh"
# file_name = "box_5x5_v4_q3_periodic.msh"
# file_name = "box_10x10.msh"
# file_name = "box_20x20.msh"
# file_name = "box_40x40.msh"
subsubfolder = "Unstructured/"
# file_name = "box_5x5_v4.msh"
file_name = "box_5x5_v4_q3_periodic.msh"
MeshFile = os.path.dirname(os.path.abspath(__file__)) + "/" + folder + subfolder + subsubfolder + file_name
mesh = MeshGmsh.import_gmsh_mesh(MeshFile)
RandomizeNodes(mesh)
RandomizeElements(mesh)
ax = np.arange(2)
ay = np.arange(2)
np.random.shuffle(ax)
np.random.shuffle(ay)
bnames_x = ["x1", "x2"]
bnames_y = ["y1", "y2"]
MeshTools.make_periodic_translational(mesh, x1=bnames_x[ax[0]], x2=bnames_x[ax[1]], y1=bnames_y[ay[0]], y2=bnames_y[ay[1]])

# MeshTools.RandomizeNodes(mesh)

### Solver parameters
dt = 0.05
FinalTime = 1.0
NumTimeSteps = int(FinalTime/dt + 10.*general.eps)
SolutionOrder = 2
Params = general.SetSolverParams(SolutionOrder=SolutionOrder,FinalTime=FinalTime,NumTimeSteps=NumTimeSteps,
								 SolutionBasis=SolutionBasis,TimeStepper="RK4",L2InitialCondition=True,
								 ElementQuadrature="Dunavant")


### Physics
physics = Euler.Euler2D(Params["SolutionOrder"], Params["SolutionBasis"], mesh)
# physics.set_physical_params(GasConstant=1.,SpecificHeatRatio=1.4)
physics.set_physical_params(GasConstant=1.,SpecificHeatRatio=1.4)
physics.set_conv_num_flux("LaxFriedrichs")
# Initial conditions
physics.set_IC(IC_type="IsentropicVortex")
# Exact solution
physics.set_exact(exact_type="IsentropicVortex")
# physics.ExactSoln.Set(Function=physics.FcnIsentropicVortexPropagation)
# Boundary conditions
# physics.set_BC(bname="wall", BC_type="StateAll", fcn_type="IsentropicVortex")
# physics.set_BC(bname="x1", BC_type="StateAll", fcn_type="IsentropicVortex")
# physics.set_BC(bname="x2", BC_type="StateAll", fcn_type="IsentropicVortex")
# physics.set_BC(bname="y1", BC_type="StateAll", fcn_type="IsentropicVortex")
# physics.set_BC(bname="y2", BC_type="StateAll", fcn_type="IsentropicVortex")
# physics.SetBC("wall",Function=physics.FcnIsentropicVortexPropagation, BCType=physics.BCType["StateAll"])
# raise Exception


### Solve
solver = Solver.DG(Params,physics,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.get_error(mesh, physics, solver, "Density")
# Plot
axis = None
# axis = [-5., 5., -5., 5.]
# Plot.prepare_plot(axis=axis, linewidth=0.5)
# Plot.PlotSolution(mesh, physics, solver, "Density", Equidistant=True, PlotExact=False, include_mesh=True, 
# 	Regular2D=True, show_triangulation=False, show_elem_IDs=True)
# Plot.save_figure(file_name=CurrentDir+'vortex', FileType='pdf', CropLevel=2)
# Plot.prepare_plot(close_all=False, linewidth=1.5)
# Plot.plot_line_probe(mesh, physics, solver, "Density", xy1=[-5.,1.], xy2=[5.,1.], nPoint=101, PlotExact=True, PlotIC=True)
# Plot.save_figure(file_name=CurrentDir+'line', FileType='pdf', CropLevel=2)
# Plot.show_plot()

# U = physics.U.Arrays[0]
# code.interact(local=locals())
