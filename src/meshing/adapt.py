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

    # Calculate adaption criterion for each element
    e = np.empty(mesh.num_elems)
    for elem in mesh.elements:
        # Get norm of gradient of each state variable at each quadrature point
        gU = evaluate_gradient_norm(
                solver.elem_operators.basis_phys_grad_elems[elem.id],
                physics.U[elem.id,:,:])
        # Get maximum density gradient
        grho = np.max(gU[:,0])
        # Multiply by mesh area
        e[elem.id] = solver.elem_operators.vol_elems[elem.id] * grho
    # Normalize the adaption criterion so it ranges from 0 to 1
    e /= np.max(e)

    # Print elements with high adaption criterion
    for elem in mesh.elements:
        if getattr(elem, 'deactivated', False): continue
        if e[elem.id] > .4 : print(elem.id, e[elem.id])

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
    print(mesh.num_elems)
    # For wedge case
    #if mesh.num_elems == 3: split_id = 1
    #elif mesh.num_elems == 7: split_id = 4
    #elif mesh.num_elems == 11: split_id = 7
    #else: split_id = 0
    # For box test
    #split_id = 0
    # For isentropic vortex
    #if mesh.num_elems == 50: split_id = [12,17,18,13]
    #else: split_id = 12

    split_id = np.argwhere(e > .3)
    needs_refinement[split_id] = True

    min_volume = .01

    # Loop over all elements
    for elem_id in range(mesh.num_elems):
        # Only refine elements that need refinement
        if needs_refinement[elem_id]:

            # Get element
            elem = mesh.elements[elem_id]

            # If the element volume is below a threshold, skip it
            if solver.elem_operators.vol_elems[elem.id] < min_volume: continue

            # Skip deactivated elements
            # TODO: add deactivated to constructor of Element?
            if getattr(elem, 'deactivated', False): continue

            # -- Figure out what face to split -- #
            # Get info about the longest face of the element
            long_face, long_face_node_ids = find_longest_face(elem, mesh)
            # Get the midpoint of this face
            midpoint = np.mean(mesh.node_coords[long_face_node_ids], axis=0)
            # Add the midpoint as a new mesh node
            mesh.node_coords = np.append(mesh.node_coords, [midpoint], axis=0)
            midpoint_id = np.size(mesh.node_coords, axis=0) - 1
            # Find which node on the long face is most counterclockwise (since
            # nodes must be ordered counterclockwise)
            ccwise_node_id, cwise_node_id = find_counterclockwise_node(elem.node_ids,
                    long_face_node_ids[0], long_face_node_ids[1])
            # Get neighbor across this face, if there is one
            neighbor_id = elem.face_to_neighbors[long_face]
            if neighbor_id != -1:
                neighbor = mesh.elements[neighbor_id]
                # Find the local ID of the long face on the neighbor
                neighbor_opposing_node_id = np.setdiff1d(neighbor.node_ids, long_face_node_ids, assume_unique=True)[0]
                neighbor_long_face = np.argwhere(neighbor.node_ids == neighbor_opposing_node_id)[0,0]

            # Split this element
            new_elem1, new_elem2 = split_element(mesh, physics, solver.basis, elem, long_face,
                    midpoint_id, ccwise_node_id, cwise_node_id)
            # Split the neighbor, making sure to flip clockwise/counterclockwise
            if neighbor_id != -1:
                new_elem3, new_elem4 = split_element(mesh, physics, solver.basis, neighbor, neighbor_long_face,
                        midpoint_id, cwise_node_id, ccwise_node_id)

            # Create the faces between the elements
            append_face(mesh, new_elem2, new_elem1)
            if neighbor_id != -1:
                append_face(mesh, new_elem3, new_elem2)
                append_face(mesh, new_elem4, new_elem3)
                append_face(mesh, new_elem1, new_elem4)

            # TODO: Figure out how to remove long face after making new ones
            # "Deactivate" the original element and its neighbor
            # TODO: figure out a better way to do this
            elem.face_to_neighbors = np.array([-1,-1,-1])
            if neighbor_id != -1:
                neighbor.face_to_neighbors = np.array([-1,-1,-1])
            offset = 1
            corner = np.array([-10,-10])
            elem.node_coords = np.array([corner, corner+[0,-offset], corner+[offset,0]])
            if neighbor_id != -1:
                neighbor.node_coords = np.array([corner+[0,-offset], corner+[offset,-offset], corner+[offset,0]])

            # Call compute operators
            solver.precompute_matrix_operators()
            if solver.limiter is not None: solver.limiter.precompute_operators(solver)

            # Delete residual
            stepper.R = np.zeros_like(physics.U)

            elem.deactivated = True
            if neighbor_id != -1: neighbor.deactivated = True

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
            elem, split_face - 2, split_face, 2)
    # Create second element
    new_elem2 = append_element(mesh, physics, basis, node_ids_2, ref_nodes_2,
            elem, split_face - 1, split_face, 1)
    return new_elem1, new_elem2

def append_element(mesh, physics, basis, node_ids, ref_nodes, parent, parent_face_id, split_face_id, new_split_face_id):
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
    new_split_face_id - local ID of face in new element which has been split
    Returns:
    elem - Element object (meshing/meshbase.py), newly created element
    """
    # The index of node_ids which contains the midpoint of the split face is
    # always 0
    midpoint_index = 0
    # Local ID of face in new element which needs new neighbors is always 0,
    # since this face always opposes the midpoint, which is node 0
    face_id = 0
    # Wrap parent face ID to be positive
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
    # If the parent's neighbor is a boundary, add a new boundary face
    if parent_neighbor_id == -1:
        # Search for the correct boundary group
        found = False
        for bgroup in mesh.boundary_groups.values():
            for bface in bgroup.boundary_faces:
                # If found, stop the search
                if bface.elem_id == parent.id and bface.face_id == parent_face_id:
                    found = True
                    break
            if found:
                bgroup.boundary_faces.append(mesh_defs.BoundaryFace())
                bgroup.boundary_faces[-1].elem_id = elem.id
                bgroup.boundary_faces[-1].face_id = face_id
                bgroup.num_boundary_faces += 1
    # If the split face is a boundary, add a new boundary face
    split_face_neighbor_id = parent.face_to_neighbors[split_face_id]
    if split_face_neighbor_id == -1:
        # Search for the correct boundary group
        found = False
        for bgroup in mesh.boundary_groups.values():
            for bface in bgroup.boundary_faces:
                # If found, stop the search
                if bface.elem_id == parent.id and bface.face_id == split_face_id:
                    found = True
                    break
            if found:
                bgroup.boundary_faces.append(mesh_defs.BoundaryFace())
                bgroup.boundary_faces[-1].elem_id = elem.id
                bgroup.boundary_faces[-1].face_id = new_split_face_id
                bgroup.num_boundary_faces += 1
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
        # Get the neighbor across the face
        face_neighbor = mesh.elements[elem.face_to_neighbors[i]]
        # Calculate the face area and find face nodes
        face_areas[i], face_node_ids[i,:] = face_geometry(elem, i,
                mesh.node_coords)
    # Get face with highest area
    long_face = np.argmax(face_areas)
    # Get node IDs of the longest face
    long_face_node_ids = face_node_ids[long_face,:]
    return (long_face, long_face_node_ids)

def face_geometry(elem, face_id, node_coords):
    """Find the area and nodes of a face.

    Arguments:
    elem - Element object (meshing/meshbase.py)
    face_id - ID of face to get area/nodes of
    node_coords - array of node coordinates, shape [num_nodes, dim]
    Returns:
    (float, array[2]) tuple of face area and node IDs
    """
    # Find the node IDs of the face. This works by removing the node
    # opposite the face (only works for first-order triangles).
    # TODO: replace this with gbasis.get_local_face_node_nums
    face_node_ids = np.setdiff1d(elem.node_ids, elem.node_ids[face_id])
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

def p1_nodes(a, b, c):
    """Find nodes of a P1 element.

    Arguments:
    a - coordinates of first corner
    b - coordinates of second corner
    c - coordinates of third corner
    Returns:
    nodes - array, shape(3,2), containing P2 node coordinates
    """
    nodes = np.empty((3,2))
    nodes[0,:] = a
    nodes[1,:] = b
    nodes[2,:] = c
    return nodes

def p0_nodes(a, b, c):
    """Find nodes of a P0 element.

    Arguments:
    a - coordinates of first corner
    b - coordinates of second corner
    c - coordinates of third corner
    Returns:
    nodes - array, shape(3,2), containing P2 node coordinates
    """
    nodes = np.empty((1,2))
    nodes[0,:] = (a+b+c)/3
    return nodes

def evaluate_gradient_norm(basis_val_grad, Uc):
    """Evaluate the norm of the gradient at a set of points within an element.

    Arguments:
    basis_val_grad - array[nq,nb,dim] of basis gradients at each point for each
            basis for each dimension
    Uc - array[nb,ns] of element solution at each node
    Returns:
    array[nq,ns] of norm of gradient of each state variable at each point
    """
    gradU = np.empty((basis_val_grad.shape[0], Uc.shape[1], basis_val_grad.shape[2]))
    # Loop over each dimension
    for i in range(basis_val_grad.shape[2]):
        gradU[:,:,i] = np.matmul(basis_val_grad[:,:,i], Uc)
    # Return norm of gradient
    return np.linalg.norm(gradU, axis=2)
