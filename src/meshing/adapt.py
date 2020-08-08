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

    # Array of flags for which elements to be split
    needs_refinement = np.zeros(mesh.num_elems, dtype=bool)
    # Just split element 0
    needs_refinement[0] = True

    # Loop over all elements
    for elem_id in range(mesh.num_elems):
        # Only refine elements that need refinement
        if needs_refinement[elem_id]:

            # TODO: Find the neighbors
            # 1. create faces, and add them to mesh.interior_faces
            # 2. assign elemL_id, elemR_id, faceL_id, faceR_id
            # 3. set elemL.face_to_neighbors[faceL_id] = elemR_id
            #    and elemR.face_to_neighbors[faceR_id] = elemL_id

            # Get element
            elem = mesh.elements[elem_id]
            # Arrays for area of each face and face nodes
            face_areas = np.empty(mesh.gbasis.NFACES)
            face_nodes = np.empty((mesh.gbasis.NFACES, 2, mesh.dim))
            # Loop over each face
            for i in range(mesh.gbasis.NFACES):
                # If it's a boundary, skip it
                if elem.face_to_neighbors[i] == -1: continue
                # Get the neighbor across the face
                neighbor = mesh.elements[elem.face_to_neighbors[i]]
                # Calculate the face area and find face nodes
                face_areas[i], face_nodes[i,:] = face_geometry_between(elem, neighbor, mesh.node_coords)
                print("face nodes:")
                print(face_areas[i])
            # Get face with highest area
            long_face = np.argmax(face_areas)
            # Get the midpoint of this face
            midpoint = np.mean(face_nodes[long_face,:,:], axis=0)

            # Add new mesh node
            mesh.node_coords = np.append(mesh.node_coords, [[-.5,0]], axis=0)
            # Add new element
            mesh.elements.append(mesh_defs.Element(4))
            # Set element's node ids, coords, and neighbors
            mesh.elements[4].node_ids = np.array([0,1,6])
            mesh.elements[4].node_coords = mesh.node_coords[mesh.elements[4].node_ids]
            mesh.elements[4].face_to_neighbors = np.full(mesh.gbasis.NFACES, -1)
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

            # -- Update solution -- #
            # TODO: map solution from old elements to new split elements
            # Append to the end of U
            physics.U = np.append(physics.U, [physics.U[-1,:,:]], axis=0)
            # Delete residual
            stepper.R = None

    # Just printing random things to check them out
    print(mesh.node_coords)
    print(mesh.interior_faces[0].elemL_id)
    print(mesh.interior_faces[0].elemR_id)
    print(mesh.interior_faces[0].faceL_id)
    print(mesh.interior_faces[0].faceR_id)
    print(mesh.boundary_groups)
    for i in range(mesh.num_elems):
            print(mesh.elements[i].id)
            print(mesh.elements[i].node_ids)
            print(mesh.elements[i].node_coords)
            print(mesh.elements[i].face_to_neighbors)

# TODO: If elem1 and elem2 are not actually neighbors, bad things will happen!
def face_geometry_between(elem1, elem2, node_coords):
    """Find the area and nodes of the face shared by two elements.

    Arguments:
    elem1 - first element object (meshing/meshbase.py)
    elem2 - second element object (meshing/meshbase.py)
    node_coords - array of node coordinates, shape [num_nodes, dim]
    Returns:
    (float, array[2,dim]) tuple of face area and node coordinates
    """
    # Find the node IDs of the face. This works by finding which two nodes
    # appear in both the current element and the neighbor across the face.
    face_node_ids = np.intersect1d(elem1.node_ids, elem2.node_ids)
    # Get the coordinates of these nodes
    face_nodes = node_coords[face_node_ids,:]
    # Return the area of the face (which is just the distance since this is a 2D
    # code)
    return (np.linalg.norm(face_nodes[0,:] - face_nodes[1,:]), face_nodes)
