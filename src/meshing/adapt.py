import numpy as np
import meshing.meshbase as mesh_defs
import meshing.tools as mesh_tools
import numerics.helpers.helpers as numerics_helpers
import random

# TODO: This whole thing only works for triangles. Generalize this.
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
    # Split a random element within a range
    closest_distance = 9e99
    ref_range = 2
    point = np.array([0,0]) + np.array([random.random()*ref_range - ref_range/2, random.random()*ref_range - ref_range/2])
    for elem in mesh.elements:
        distance = np.linalg.norm(np.mean(elem.node_coords, axis=0) - point)
        if distance < closest_distance:
            closest_distance = distance
            closest_elem = elem
    split_id = closest_elem.id
    if mesh.num_elems == 50: split_id = [37,17,13,18]
    #elif mesh.num_elems == 54: split_id = [50,51,52,53]
    else: split_id = 37
    needs_refinement[split_id] = True

    # Loop over all elements
    for elem_id in range(mesh.num_elems):
        # Only refine elements that need refinement
        if needs_refinement[elem_id]:

            # Get element
            elem = mesh.elements[elem_id]

            # Skip deactivated elements
            # TODO: add deactivated to constructor of Element?
            if getattr(elem, 'deactivated', False): continue

            # -- Figure out what face to split -- #
            # Get info about the longest face of the element
            long_face, long_face_node_ids = find_longest_face(elem, mesh)
            # Get neighbor across this face
            neighbor = mesh.elements[elem.face_to_neighbors[long_face]]
            # Get the midpoint of this face
            midpoint = np.mean(mesh.node_coords[long_face_node_ids], axis=0)
            # Add the midpoint as a new mesh node
            mesh.node_coords = np.append(mesh.node_coords, [midpoint], axis=0)
            midpoint_id = np.size(mesh.node_coords, axis=0) - 1
            # Find which node on the long face is most counterclockwise (since
            # nodes must be ordered counterclockwise)
            ccwise_node_id, cwise_node_id = find_counterclockwise_node(elem.node_ids,
                    long_face_node_ids[0], long_face_node_ids[1])
            # Find the local ID of the long face on the neighbor
            neighbor_opposing_node_id = np.setdiff1d(neighbor.node_ids, long_face_node_ids, assume_unique=True)[0]
            neighbor_long_face = np.argwhere(neighbor.node_ids == neighbor_opposing_node_id)[0,0]

            # Split this element
            new_elem1, new_elem2 = split_element(mesh, physics, solver.basis, elem, long_face,
                    midpoint_id, ccwise_node_id, cwise_node_id)
            # Split the neighbor, making sure to flip clockwise/counterclockwise
            new_elem3, new_elem4 = split_element(mesh, physics, solver.basis, neighbor, neighbor_long_face,
                    midpoint_id, cwise_node_id, ccwise_node_id)

            # Create the faces between the elements
            append_face(mesh, new_elem2, new_elem1)
            append_face(mesh, new_elem3, new_elem2)
            append_face(mesh, new_elem4, new_elem3)
            append_face(mesh, new_elem1, new_elem4)

            # TODO: Figure out how to remove long face after making new ones
            # "Deactivate" the original element and its neighbor
            # TODO: figure out a better way to do this
            elem.face_to_neighbors = np.array([-1,-1,-1])
            neighbor.face_to_neighbors = np.array([-1,-1,-1])
            offset = 1
            corner = np.array([-10,-10])
            elem.node_coords = np.array([corner, corner+[0,-offset], corner+[offset,0]])
            neighbor.node_coords = np.array([corner+[0,-offset], corner+[offset,-offset], corner+[offset,0]])
            #for iface in mesh.interior_faces:
            #    if (iface.elemL_id == elem.id or iface.elemR_id == elem.id
            #            or iface.elemL_id == neighbor.id or iface.elemR_id == neighbor.id):
            #        print("found face:")
            #        print(iface.elemL_id)
            #        print(iface.elemR_id)

            # Call compute operators
            solver.precompute_matrix_operators()

            # -- Update solution -- #
            # TODO: map solution from old elements to new split elements
            # Append to the end of U
            #physics.U = np.append(physics.U, [physics.U[-1,:,:]], axis=0)
            #physics.U = np.append(physics.U, [physics.U[-1,:,:]], axis=0)
            #physics.U = np.append(physics.U, [physics.U[-1,:,:]], axis=0)
            #physics.U = np.append(physics.U, [physics.U[-1,:,:]], axis=0)
            # Delete residual
            stepper.R = np.zeros_like(physics.U)

            elem.deactivated = True
            neighbor.deactivated = True

            #basis_val = solver.basis.get_values(np.array([[-1,-1]]))
            #print(basis_val)
            #print(elem.quad_pts)

            #Uq = numerics_helpers.evaluate_state(Up, basis_val)

    # Just printing things to check them out
    #print(mesh.node_coords)
    #for i in range(mesh.num_interior_faces):
    #    print("Face ", i)
    #    print(mesh.interior_faces[i].elemL_id)
    #    print(mesh.interior_faces[i].elemR_id)
    #    print(mesh.interior_faces[i].faceL_id)
    #    print(mesh.interior_faces[i].faceR_id)
    #print(mesh.boundary_groups)
    #for i in range(mesh.num_elems):
    #        print(mesh.elements[i].id)
    #        print(mesh.elements[i].node_ids)
    #        print(mesh.elements[i].node_coords)
    #        print(mesh.elements[i].face_to_neighbors)


def split_element(mesh, physics, basis, elem, split_face, midpoint_id, ccwise_node_id, cwise_node_id):
    """Split an element into two smaller elements.

    Arguments:
    mesh - Mesh object (meshing/meshbase.py)
    physics - PhysicsBase object (physics/base/base.py)
    basis - BasisBase object (numerics/basis/basis.py)
    elem - Element object (meshing/meshbase.py) that needs to be split
    split_face - local ID of face along which to split
    midpoint_id - global ID of the node which is the midpoint of the split face
    Returns:
    (new_elem1, new_elem2) - tuple of newly created Element objects
    """

    # Node coords of ref element
    ref_nodes = mesh.gbasis.get_nodes(mesh.gbasis.order)
    # Nodes of new elements in reference element
    ref_ccwise_node_id = (split_face - 1) % mesh.gbasis.nb
    ref_cwise_node_id  = (split_face + 1) % mesh.gbasis.nb
    ref_midpoint = np.mean(ref_nodes[[ref_ccwise_node_id, ref_cwise_node_id],:], axis=0)
    # Nodes of new elements in parent reference
    # TODO: Write a more general way to get high-order split elements in parent
    # reference space - this assumes P2 elements!
    ref_nodes_1 = p2_nodes(ref_midpoint, ref_nodes[ref_ccwise_node_id], ref_nodes[split_face])
    ref_nodes_2 = p2_nodes(ref_midpoint, ref_nodes[split_face], ref_nodes[ref_cwise_node_id])
    # Global node IDs of new elements
    node_ids_1 = np.array([midpoint_id, ccwise_node_id, elem.node_ids[split_face]])
    node_ids_2 = np.array([midpoint_id, elem.node_ids[split_face], cwise_node_id])

    # Create first element
    new_elem1 = append_element(mesh, physics, basis, node_ids_1, ref_nodes_1,
            elem, split_face - 2, split_face)
    # Create second element
    new_elem2 = append_element(mesh, physics, basis, node_ids_2, ref_nodes_2,
            elem, split_face - 1, split_face)
    return new_elem1, new_elem2

def append_element(mesh, physics, basis, node_ids, ref_nodes, parent, parent_face_id, split_face_id):
    """Create a new element at specified nodes and append it to the mesh.
    This function creates a new element and sets the neighbors of the element
    and neighbor element across the face specified by face_id. The solution
    array is interpolated from the parent element to the two new elements.

    Arguments:
    mesh - Mesh object (meshing/meshbase.py)
    physics - PhysicsBase object (physics/base/base.py)
    basis - BasisBase object (numerics/basis/basis.py)
    node_ids - array of new element's node IDs
    parent - Element object (meshing/meshbase.py), parent of the new element
    parent_face_id - local ID of face in parent element which needs new neighbors
    split_face_id - local ID of face in parent element which has been split
    Returns:
    elem - Element object (meshing/meshbase.py), newly created element
    """
    # The index of node_ids which contains the midpoint of the split face is
    # always 0
    midpoint_index = 0
    # Local ID of face in new element which needs new neighbors is always 0,
    # since this face always opposes the midpoint, which is node 0
    face_id = 0
    # Wrap parent face ID index to be positive
    parent_face_id = parent_face_id % mesh.gbasis.NFACES
    # Create element
    mesh.elements.append(mesh_defs.Element())
    elem = mesh.elements[-1]
    # Set first element's id, node ids, coords, and neighbors
    elem.id = len(mesh.elements) - 1
    elem.node_ids = node_ids
    elem.node_coords = mesh.node_coords[elem.node_ids]
    elem.face_to_neighbors = np.full(mesh.gbasis.NFACES, -1)
    # Append to element nodes in mesh
    mesh.elem_to_node_ids = np.append(mesh.elem_to_node_ids, [node_ids], axis=0)
    # Update number of elements
    mesh.num_elems += 1
    # Get parent's neighbor across face
    parent_neighbor_id = parent.face_to_neighbors[parent_face_id]
    # Add parent's neighbor to new elements's neighbor
    elem.face_to_neighbors[face_id] = parent_neighbor_id
    # If the parent's neighbor is not a boundary, update it
    if parent_neighbor_id != -1:
        # Get parent's neighbor
        parent_neighbor = mesh.elements[parent_neighbor_id]
        # Get index of face in parent's neighbor
        parent_neighbor_face_index = np.argwhere(parent_neighbor.face_to_neighbors == parent.id)[0]
        # Set new element as parent neighbor's neighbor
        parent_neighbor.face_to_neighbors[parent_neighbor_face_index] = elem.id
        # Update old face neighbors by looking for the face between parent and parent_neighbor
        for face in mesh.interior_faces:
            if face.elemL_id == parent.id and face.elemR_id == parent_neighbor.id:
                face.elemL_id = elem.id
                face.faceL_id = face_id
                break
            if face.elemR_id == parent.id and face.elemL_id == parent_neighbor.id:
                face.elemR_id = elem.id
                face.faceR_id = face_id
                break
    # If the parent's neighbor is a boundary, update the boundary
    if parent_neighbor_id == -1:
        # Search for the correct boundary face
        found = False
        for bgroup in mesh.boundary_groups.values():
            for bface in bgroup.boundary_faces:
                # If found, change it to neighbor the new element
                if bface.elem_id == parent.id and bface.face_id == parent_face_id:
                    bface.elem_id = elem.id
                    bface.face_id = face_id
                    found = True
                    break
            if found: break
    # -- Set nodal solution values of new element -- #
    # Evaluate basis functions at new nodes on parent reference element
    basis_vals = basis.get_values(ref_nodes)
    # Evaluate the state at these new nodes, and append to global solution
    U_elem = numerics_helpers.evaluate_state(physics.U[parent.id,:,:], basis_vals)
    physics.U = np.append(physics.U, [U_elem], axis=0)
    return elem

def append_face(mesh, elemL, elemR):
    """Create a new face between two elements and append it to the mesh.

    Arguments:
    mesh - Mesh object (meshing/meshbase.py)
    elemL - Element object, left of face (meshing/meshbase.py)
    elemR - Element object, right of face (meshing/meshbase.py)
    """
    # Create the face between the elements
    mesh.interior_faces.append(mesh_defs.InteriorFace())
    mesh.interior_faces[-1].elemL_id = elemL.id
    mesh.interior_faces[-1].elemR_id = elemR.id
    # This assumes midpoint is node 0
    mesh.interior_faces[-1].faceL_id = 2
    mesh.interior_faces[-1].faceR_id = 1
    # Set neighbors on either side of the face
    elemL.face_to_neighbors[2] = elemR.id
    elemR.face_to_neighbors[1] = elemL.id
    # Update number of faces
    mesh.num_interior_faces += 1

def find_longest_face(elem, mesh):
    """Find the longest face in an element.

    Arguments:
    elem - Element object (meshing/meshbase.py)
    mesh - Mesh object (meshing/meshbase.py)
    Returns:
    (int, array[2]) - tuple of longest face ID and array of node IDs
    """
    # Arrays for area of each face and face nodes
    face_areas = np.zeros(mesh.gbasis.NFACES)
    face_node_ids = np.empty((mesh.gbasis.NFACES, 2), dtype=int)
    # Loop over each face
    for i in range(mesh.gbasis.NFACES):
        # If it's a boundary, skip it
        if elem.face_to_neighbors[i] == -1: continue
        # Get the neighbor across the face
        face_neighbor = mesh.elements[elem.face_to_neighbors[i]]
        # Calculate the face area and find face nodes
        face_areas[i], face_node_ids[i,:] = face_geometry_between(elem,
                face_neighbor, mesh.node_coords)
    # Get face with highest area
    long_face = np.argmax(face_areas)
    # Get node IDs of the longest face
    long_face_node_ids = face_node_ids[long_face,:]
    return (long_face, long_face_node_ids)

# TODO: If elem1 and elem2 are not actually neighbors, bad things will happen!
def face_geometry_between(elem1, elem2, node_coords):
    """Find the area and nodes of the face shared by two elements.

    Arguments:
    elem1 - first Element object (meshing/meshbase.py)
    elem2 - second Element object (meshing/meshbase.py)
    node_coords - array of node coordinates, shape [num_nodes, dim]
    Returns:
    (float, array[2]) tuple of face area and node IDs
    """
    # Find the node IDs of the face. This works by finding which two nodes
    # appear in both the current element and the neighbor across the face.
    face_node_ids = np.intersect1d(elem1.node_ids, elem2.node_ids)
    # Get the coordinates of these nodes
    face_nodes = node_coords[face_node_ids,:]
    # Return the area of the face (which is just the distance since this is a 2D
    # code)
    return (np.linalg.norm(face_nodes[0,:] - face_nodes[1,:]), face_node_ids)

def find_counterclockwise_node(nodes, a, b):
    """Find which of two neighboring nodes is more counterclockwise.
    The function takes an array of nodes along with two neighboring nodes a and b, then
    finds which of a and b are more counterclockwise. The nodes array is assumed
    to be ordered counterclockwise, which means that whichever of a and b
    appears later in the array (or appears at index 0) is the most counterclockwise.

    Arguments:
    nodes - array of node IDs
    a - ID of the first node
    b - ID of the second node
    Returns
    (int, int) tuple of counterclockwise node and clockwise node
    """

    # Find indices of a and b in the nodes array
    a_index = np.argwhere(nodes == a)[0]
    b_index = np.argwhere(nodes == b)[0]

    # If a's index is higher
    if a_index > b_index:
        # If a is at the end and b is at the beginning, then b is ahead
        if a_index == nodes.size-1 and b_index == 0:
            ccwise_node = b
            cwise_node = a
        # In every other case, if a's index is higher then a is ahead
        else:
            ccwise_node = a
            cwise_node = b
    # If b's index is higher
    else:
        # If b is at the end and a is at the beginning, then a is ahead
        if b_index == nodes.size-1 and a_index == 0:
            ccwise_node = a
            cwise_node = b
        # In every other case, if b's index is higher then b is ahead
        else:
            ccwise_node = b
            cwise_node = a
    return (ccwise_node, cwise_node)

def p2_nodes(a, b, c):
    """Find nodes of a P2 element.

    Arguments:
    a - coordinates of first corner
    b - coordinates of second corner
    c - coordinates of third corner
    Returns:
    nodes - array, shape(6,2), containing P2 node coordinates
    """
    nodes = np.empty((6,2))
    nodes[0,:] = a
    nodes[1,:] = np.mean([a,b],axis=0)
    nodes[2,:] = b
    nodes[3,:] = np.mean([b,c],axis=0)
    nodes[4,:] = c
    nodes[5,:] = np.mean([c,a],axis=0)
    return nodes
