import numpy as np
import meshing.meshbase as mesh_defs

def adapt(solver, physics, mesh, stepper):
    """Adapt the mesh by refining or coarsening it.
    For now, this does very little - just splits an element in half.

    Arguments:
    solver - Solver object (solver/base.py)
    physics - Physics object (physics/base.py)
    mesh - Mesh object (meshing/meshbase.py)
    stepper - Stepper object (timestepping/stepper.py)
    """

    # Append to the end of U
    physics.U = np.append(physics.U, [physics.U[-1,:,:]], axis=0)
    # Delete residual
    stepper.R = None
    print("appended U:")
    print(physics.U)
    # Add new mesh node
    mesh.node_coords = np.append(mesh.node_coords, [[-.5,0]], axis=0)
    # Add new element
    mesh.elements.append(mesh_defs.Element(4))
    # Set element's node ids, coords, and neighbors
    mesh.elements[4].node_ids = np.array([0,1,6])
    mesh.elements[4].node_coords = np.array([[-1., -1.], [ 0., -1.], [-.5, 0.]])
    mesh.elements[4].face_to_neighbors = np.array([-1,-1,-1])
    # Make the first element smaller to make space
    mesh.elements[0].node_ids = np.array([0,6,3])
    mesh.elements[0].node_coords = np.array([[-1., -1.], [ -.5, 0.], [-1., 1.]])
    mesh.elements[0].face_to_neighbors = np.array([-1,-1,-1])
    # Increment number of elements
    mesh.num_elems += 1
    print("num_elems: ", mesh.num_elems)
    solver.elem_operators.x_elems = np.append(solver.elem_operators.x_elems, [solver.elem_operators.x_elems[-1,:,:]], axis=0)
    # Call compute operators
    solver.precompute_matrix_operators()
    print(solver.elem_operators.x_elems)

    # Just printing random things to check them out
    print(mesh.node_coords)
    print(mesh.IFaces[0].elemL_id)
    print(mesh.IFaces[0].elemR_id)
    print(mesh.IFaces[0].faceL_id)
    print(mesh.IFaces[0].faceR_id)
    print(mesh.BFaceGroups)
    for i in range(mesh.num_elems):
            print(mesh.elements[i].id)
            print(mesh.elements[i].node_ids)
            print(mesh.elements[i].node_coords)
            print(mesh.elements[i].face_to_neighbors)
