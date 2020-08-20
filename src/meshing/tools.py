import code
import numpy as np 

import data

import meshing.meshbase as mesh_defs

import numerics.basis.tools as basis_tools

TOL = 1.e-10


def ref_to_phys(mesh, elem_id, xref):
    '''
    This function converts reference space coordinates to physical
    space coordinates

    Inputs:
    -------
        mesh: mesh object
        elem_id: element ID
        xref: coordinates in reference space [nq, dim]

    Outputs:
    --------
        xphys: coordinates in physical space [nq, dim]
    '''
    gbasis = mesh.gbasis
    gorder = mesh.gorder

    # Get basis values
    gbasis.get_basis_val_grads(xref, get_val=True)

    # Element node coordinates
    elem_coords = mesh.elements[elem_id].node_coords

    # Convert to physical space
    xphys = np.matmul(gbasis.basis_val, elem_coords)

    return xphys


def element_volumes(mesh, solver=None):
    '''
    This function calculates total and per-element volumes

    Inputs:
    -------
        mesh: mesh object
        solver: solver object (e.g., DG, ADER-DG, etc.)
    
    Outputs:
    --------
        vol_elems: volume of each element [num_elems]
        domain_vol: total volume of the domain
    '''
    # Check if already calculated
    if solver is not None:
        if hasattr(solver.elem_operators, "domain_vol") \
                and hasattr(solver.elem_operators, "vol_elems"):
            return solver.elem_operators.vol_elems, \
                    solver.elem_operators.domain_vol

    # Allocate, unpack
    vol_elems = np.zeros(mesh.num_elems)
    gorder = mesh.gorder
    gbasis = mesh.gbasis

    # Get quadrature data
    quad_order = gbasis.get_quadrature_order(mesh, gorder)
    quad_pts, quad_wts = gbasis.get_quadrature_data(quad_order)

    # Get element volumes
    for elem_id in range(mesh.num_elems):
        djac, _, _ = basis_tools.element_jacobian(mesh, elem_id, quad_pts, 
                get_djac=True)
        vol_elems[elem_id] = np.sum(quad_wts*djac)

    # Get domain volume
    domain_vol = np.sum(vol_elems)

    return vol_elems, domain_vol


def get_element_centroid(mesh, elem_id):
    '''
    This function obtains the centroid of an element in physical space.

    Inputs:
    -------
        mesh: mesh object
        elem_ID: element ID
    
    Outputs:
    --------
        xcentroid: element centroid in physical space [1, dim]
    '''
    gbasis = mesh.gbasis
    xcentroid = ref_to_phys(mesh, elem_id, mesh.gbasis.CENTROID)  

    return xcentroid


def check_face_orientations(mesh):
    '''
    This function checks the face orientations for 2D meshes.

    Inputs:
    -------
        mesh: mesh object

    Notes:
    ------
        An error is raised if face orientations don't match up.
    '''
    gbasis = mesh.gbasis
    if mesh.dim == 1:
        # don't need to check for 1D
        return

    for interior_face in mesh.interior_faces:
        elemL_id = interior_face.elemL_id
        elemR_id = interior_face.elemR_id
        faceL_id = interior_face.faceL_id
        faceR_id = interior_face.faceR_id

        # Get local IDs of element nodes
        elemL_node_ids = mesh.elements[elemL_id].node_ids
        elemR_node_ids = mesh.elements[elemR_id].node_ids

        ''' Get global IDs of face nodes '''
        # Local IDs - left
        face_node_ids = gbasis.get_local_face_principal_node_nums(
                mesh.gorder, faceL_id)
        # Global IDs - left
        global_node_ids_L = elemL_node_ids[face_node_ids]
        # Local IDs - right
        face_node_ids = gbasis.get_local_face_principal_node_nums(
                mesh.gorder, faceR_id)
        # Global IDs - right
        global_node_ids_R = elemR_node_ids[face_node_ids]

        # Node ordering should be reversed between the two elements
        if not np.all(global_node_ids_L == global_node_ids_R[::-1]):
            raise Exception("Face orientation for elemL_id = %d, elemR_id " +
                "= %d \\ is incorrect" % (elemL_id, elemR_id))


def verify_periodic_compatibility(mesh, boundary_group, icoord):
    '''
    This function checks whether a boundary is compatible with periodicity.
    Specifically, it verifies that all boundary nodes are located on the same
    plane, up to a given tolerance. It then potentially slightly modifies the
    node coordinates to ensure the exact same value.

    Inputs:
    -------
        mesh: mesh object
        boundary_group: boundary group object
        icoord: which spatial direction to check (0 for x, 1 for y)
    
    Outputs:
    --------
        mesh: mesh object (coordinates potentially modified)
        coord: position of boundary in the icoord direction
    '''
    coord = np.nan
    gbasis = mesh.gbasis
    for boundary_face in boundary_group.boundary_faces:
        # Extract info
        elem_id = boundary_face.elem_id
        face_id = boundary_face.face_id

        # Get local IDs of principal nodes on face
        # local_node_ids = gbasis.get_local_face_principal_node_nums(
        #         mesh.gorder, face_id)
        local_node_ids = gbasis.get_local_face_node_nums(
                mesh.gorder, face_id)

        # Physical coordinates of nodes
        elem_coords = mesh.elements[elem_id].node_coords
        coords = elem_coords[local_node_ids]

        # Make sure all nodes have same icoord-position (within TOL)
        if np.isnan(coord):
            coord = coords[0, icoord]
        if np.any(np.abs(coords[:, icoord] - coord) > TOL):
            raise ValueError("Boundary %s not compatible with periodicity" % 
                    (boundary_group.name))

        # Now force each node to have the same exact icoord-position
        coords[:, icoord] = coord

    return coord


def reorder_periodic_boundary_nodes(mesh, b1, b2, icoord, 
        old_to_new_node_map, new_to_old_node_map, next_node_id):
    '''
    This function checks whether a boundary is compatible with periodicity.
    Specifically, it verifies that all boundary nodes are located on the same
    plane, up to a given tolerance. It then potentially slightly modifies the
    node coordinates to ensure the exact same value.

    Inputs:
    -------
        mesh: mesh object
        b1: name of 1st periodic boundary
        b2: name of 2nd periodic boundary
        icoord: spatial direction of periodicity (0 for x, 1 for y)
        old_to_new_node_map: maps old to new node IDs [num_nodes]
        new_to_old_node_map: maps new to old node IDs [num_nodes]
        next_node_id: next new node ID to assign
    
    Outputs:
    --------
        boundary_group1: boundary group object for 1st periodic boundary
        boundary_group2: boundary group object for 2nd periodic boundary
        node_pairs: stores pairs of nodes that match each other on opposite
            boundaries [num_node_pairs, 2]
        next_node_id: next new node ID to assign (modified)
        old_to_new_node_map: maps old to new node IDs (modified) [num_nodes]
        new_to_old_node_map: maps new to old node IDs (modified) [num_nodes]

    Notes:
    ------
        node_pairs[i] = np.array([node1_id, node2_id]) is the ith node pair, 
        where node1_id is the ID of a node on boundary 1 and node2_id is the 
        ID of the node on boundary 2 that corresponds to node1
    '''
    gbasis = mesh.gbasis

    if b1 is None and b2 is None:
        # Trivial case - no periodicity in given direction
        return None, None, None, next_node_id
    elif b1 == b2:
        raise ValueError("Duplicate boundaries")

    # Allocate arrays
    node_pairs = np.zeros([mesh.num_nodes, 2], dtype=int) - 1
    idx_in_node_pairs = np.zeros(mesh.num_nodes, dtype=int) - 1
    num_node_pairs = 0
    node2_matched = np.zeros(mesh.num_nodes, dtype=bool)

    # Extract the two boundary_groups
    boundary_group1 = mesh.boundary_groups[b1]
    boundary_group2 = mesh.boundary_groups[b2]

    start_node_id = next_node_id

    if icoord < 0 or icoord >= mesh.dim:
        raise ValueError

    ''' 
    Make sure each boundary is compatible with periodicity 
    Note: the boundary node coordinates may be slightly modified
    to ensure same coordinate in periodic direction
    '''
    if boundary_group1.num_boundary_faces != \
            boundary_group2.num_boundary_faces:
        raise ValueError
    pcoord1 = verify_periodic_compatibility(mesh, boundary_group1, icoord)
    pcoord2 = verify_periodic_compatibility(mesh, boundary_group2, icoord)
    pdiff = np.abs(pcoord1-pcoord2) # distance between the two boundaries

    '''
    Deal with first boundary
    '''
    # Populate node maps for first boundary
    for boundary_face in boundary_group1.boundary_faces:
        # Extract info
        elem_id = boundary_face.elem_id
        face_id = boundary_face.face_id

        # Local IDs of face nodes
        local_node_ids = gbasis.get_local_face_principal_node_nums( 
                mesh.gorder, face_id)
        # Global IDs of face nodes
        global_node_ids = mesh.elem_to_node_ids[elem_id][local_node_ids]

        # Populate node maps
        for node_id in global_node_ids:
            if old_to_new_node_map[node_id] == -1: 
                # Node already mapped
                old_to_new_node_map[node_id] = next_node_id
                new_to_old_node_map[next_node_id] = node_id
                next_node_id += 1
            if idx_in_node_pairs[node_id] == -1:
                node_pairs[num_node_pairs, 0] = node_id
                idx_in_node_pairs[node_id] = num_node_pairs
                    # maps node ID to index in node_pairs
                num_node_pairs += 1

    # Last ID assigned to nodes on boundary 1
    stop_node_id = next_node_id

    '''
    Deal with second boundary
    '''
    # Populate node maps for second boundary
    for boundary_face in boundary_group2.boundary_faces:
        # Extract info
        elem_id = boundary_face.elem_id
        face_id = boundary_face.face_id

        # Local IDs of face nodes
        local_node_ids = gbasis.get_local_face_principal_node_nums( 
                mesh.gorder, face_id)
        # Global IDs of face nodes
        global_node_ids = mesh.elem_to_node_ids[elem_id][local_node_ids]

        for node2_id in global_node_ids:
            ''' Find matching nodes on boundary group 1 '''

            if node2_matched[node2_id]: 
                # this node already matched - skip
                # sanity check
                if old_to_new_node_map[node2_id] == -1:
                    raise ValueError("Node %d already matched" % node2_id)
                continue

            # Physical coordinates of node
            coord2 = mesh.node_coords[node2_id]
            
            match = False
            # Match with a node on boundary 1
            for n in range(num_node_pairs):
                if node_pairs[n, 1] != -1:
                    # node1 already paired - skip
                    continue

                node1_id = node_pairs[n, 0]
                coord1 = mesh.node_coords[node1_id]

                # Find distance between the two nodes
                norm = np.linalg.norm(coord1-coord2, ord=1)

                # Check if distance is equal to pdiff (within TOL)
                if np.abs(norm-pdiff) < TOL:
                    match = True
                    if old_to_new_node_map[node2_id] == -1:
                        # node2 not reordered yet

                        # Populate maps
                        # Note: the difference in IDs of matching nodes is
                        # equal to (stop_node_id - start_node_id)
                        node2_id_new = old_to_new_node_map[node1_id] + \
                                stop_node_id - start_node_id
                        old_to_new_node_map[node2_id] = node2_id_new
                        new_to_old_node_map[node2_id_new] = node2_id
                        next_node_id = np.amax([next_node_id, node2_id_new])

                        # Force nodes to match exactly
                        for d in range(mesh.dim):
                            if d == icoord: 
                                # Skip periodic direction
                                continue 
                            coord2[d] = coord1[d]

                    # Store node pair
                    idx1 = idx_in_node_pairs[node1_id]
                    node_pairs[idx1, 1] = node2_id
                    idx_in_node_pairs[node2_id] = idx1

                    # Flag node2 as matched
                    node2_matched[node2_id] = True
                    break

            if not match:
                raise ValueError("Could not find matching boundary node " +
                        "for Node %d" % (node2_id))

    # Modify next node ID
    if start_node_id != stop_node_id:
        # This means at least one pair of nodes was matched
        next_node_id += 1
    # Sanity check
    if next_node_id != 2*stop_node_id - start_node_id:
        raise ValueError

    # Resize node_pairs
    node_pairs = node_pairs[:num_node_pairs, :]

    # Print info
    if icoord == 0:
        s = "x"
    elif icoord == 1:
        s = "y"
    else:
        s = "z"
    print("Reordered periodic boundary in %s-direction" % (s))

    return boundary_group1, boundary_group2, node_pairs, next_node_id  


def remap_nodes(mesh, old_to_new_node_map, new_to_old_node_map, 
        next_node_id=-1):
    '''
    This function remaps node IDs based on the maps passed in as input
    arguments.

    Inputs:
    -------
        mesh: mesh object
        old_to_new_node_map: maps old to new node IDs [num_nodes]
        new_to_old_node_map: maps new to old node IDs [num_nodes]
        next_node_id: next new node ID to assign
    
    Outputs:
    --------
        mesh: mesh object (modified)
        old_to_new_node_map: maps old to new node IDs (modified)
        new_to_old_node_map: maps new to old node IDs (modified)
    '''

    # Fill up node maps with non-periodic nodes
    # Note: non-periodic nodes come after the periodic nodes
    if next_node_id != -1:
        for node_id in range(mesh.num_nodes):
            if old_to_new_node_map[node_id] == -1: 
                # has not been re-ordered yet
                old_to_new_node_map[node_id] = next_node_id
                new_to_old_node_map[next_node_id] = node_id
                next_node_id += 1

    # Assign new node IDs
    mesh.node_coords = mesh.node_coords[new_to_old_node_map]

    # New elem_to_node_ids
    num_elems = mesh.num_elems
    for elem_id in range(num_elems):
        mesh.elem_to_node_ids[elem_id,:] = old_to_new_node_map[
                mesh.elem_to_node_ids[elem_id, :]]


def match_boundary_pair(mesh, icoord, boundary_group1, boundary_group2, 
        node_pairs, old_to_new_node_map, new_to_old_node_map):
    '''
    This function creates interior faces to match the two periodic 
    boundaries.

    Inputs:
    -------
        mesh: mesh object
        icoord: spatial direction of periodicity (0 for x, 1 for y)
        boundary_group1: boundary object of 1st periodic boundary
        boundary_group2: boundary object of 2nd periodic boundary
        node_pairs: stores pairs of nodes that match each other on opposite
            boundaries [num_node_pairs, 2]
        old_to_new_node_map: maps old to new node IDs [num_nodes]
        new_to_old_node_map: maps new to old node IDs [num_nodes]
    
    Outputs:
    --------
        mesh: mesh object (modified - new interior faces, removed boundary 
            groups)
    '''
    gbasis = mesh.gbasis
    interior_faces = mesh.interior_faces

    if boundary_group1 is None and boundary_group2 is None:
        return
    elif boundary_group1 is None or boundary_group2 is None:
        raise ValueError("Only one boundary group provided")

    '''
    Remap node_pairs and idx_in_node_pairs
    '''
    # Allocate
    new_node_pairs = np.zeros_like(node_pairs, dtype=int) - 1
    idx_in_node_pairs = np.zeros(mesh.num_nodes, dtype=int) - 1
    # Remap
    new_node_pairs = old_to_new_node_map[node_pairs]
    n1 = new_node_pairs[:, 0]
    idx_in_node_pairs[n1] = np.arange(n1.shape[0])
    # Replace
    node_pairs = new_node_pairs

    # Sanity check
    if np.amin(new_node_pairs) == -1:
        raise ValueError

    ''' 
    Identify and create periodic interior_faces 
    '''
    for boundary_face1 in boundary_group1.boundary_faces:
        # Extract info
        elem_id1 = boundary_face1.elem_id
        face_id1 = boundary_face1.face_id

        # Local IDs of face nodes
        local_node_ids = gbasis.get_local_face_principal_node_nums(
            mesh.gorder, face_id1)
        # Global IDs of face nodes
        global_node_ids = mesh.elem_to_node_ids[elem_id1][local_node_ids]
        # Sort for easy comparison later
        global_node_ids_1 = np.sort(global_node_ids)

        # Pair each node with corresponding one on other boundary
        for boundary_face2 in boundary_group2.boundary_faces:
            # Extract info
            elem_id2 = boundary_face2.elem_id
            face_id2 = boundary_face2.face_id

            # Local IDs of face nodes
            local_node_ids = gbasis.get_local_face_principal_node_nums(
                    mesh.gorder, face_id2)
            # Global IDs of face nodes
            global_node_ids = mesh.elem_to_node_ids[elem_id2][local_node_ids]
            # Sort for easy comparison later
            global_node_ids_2 = np.sort(global_node_ids)

            ''' Check for complete match between all nodes '''
            # Get nodes on boundary 2 paired with those in global_node_ids_1
            idx1 = idx_in_node_pairs[global_node_ids_1]
            nodes1_partner_ids = node_pairs[idx1, 1]
            # Sort
            nodes1_partner_ids_sort = np.sort(nodes1_partner_ids)

            match = False
            if np.all(global_node_ids_2 == nodes1_partner_ids_sort):
                # Matching nodes
                match = True

                # Sanity check
                if not np.all(global_node_ids_2 == nodes1_partner_ids):
                    raise ValueError("Node ordering on opposite periodic " +
                            "faces is different")

                # Create interior face between these two faces
                mesh.num_interior_faces += 1
                interior_faces.append(mesh_defs.InteriorFace())
                interior_face = interior_faces[-1]
                interior_face.elemL_id = elem_id1
                interior_face.faceL_id = face_id1
                interior_face.elemR_id = elem_id2
                interior_face.faceR_id = face_id2

                # Decrement number of boundary faces
                boundary_group1.num_boundary_faces -= 1
                boundary_group2.num_boundary_faces -= 1

                break

        if not match:
            raise ValueError("Could not find matching boundary face")

    # Verification
    if boundary_group1.num_boundary_faces != 0 or \
            boundary_group2.num_boundary_faces != 0:
        raise ValueError

    # Remove the 2 boundary groups
    mesh.num_boundary_groups -= 2
    mesh.boundary_groups.pop(boundary_group1.name)
    mesh.boundary_groups.pop(boundary_group2.name)

    # Print info
    if icoord == 0:
        s = "x"
    elif icoord == 1:
        s = "y"
    else:
        s = "z"
    print("Matched periodic boundaries in %s-direction" % (s)) 


def update_boundary_group_nums(mesh):
    '''
    This function updates the boundary group numbers.

    Inputs:
    -------
        mesh: mesh object
    
    Outputs:
    --------
        mesh: mesh object (boundary group numbers modified)
    '''
    i = 0
    for boundary_group in mesh.boundary_groups.values():
        boundary_group.number = i
        i += 1


def verify_periodic_mesh(mesh):
    '''
    This function verifies the periodicity of the mesh.

    Inputs:
    -------
        mesh: mesh object
    '''
    # Loop through interior faces
    for interior_face in mesh.interior_faces:
        # Extract info
        elemL_id = interior_face.elemL_id
        elemR_id = interior_face.elemR_id
        faceL_id = interior_face.faceL_id
        faceR_id = interior_face.faceR_id
        gbasis = mesh.gbasis
        gorder = mesh.gorder

        ''' Get global IDs of face nodes '''
        # Local IDs - left
        local_node_ids = gbasis.get_local_face_principal_node_nums( 
                gorder, faceL_id)
        # Global IDs - left
        global_node_ids_L = mesh.elem_to_node_ids[elemL_id][local_node_ids]
        # Local IDs - right
        local_node_ids = gbasis.get_local_face_principal_node_nums( 
            gorder, faceR_id)
        # Global IDs - right
        global_node_ids_R = mesh.elem_to_node_ids[elemR_id][local_node_ids]

        ''' If exact same global nodes, then this is NOT a periodic face '''
        # Sort for easy comparison
        global_node_ids_L = np.sort(global_node_ids_L)
        global_node_ids_R = np.sort(global_node_ids_R)
        if np.all(global_node_ids_L == global_node_ids_R):
            # Skip non-periodic faces
            continue

        ''' Compare distances '''
        coordsL = mesh.node_coords[global_node_ids_L]
        coordsR = mesh.node_coords[global_node_ids_R]
        dists = np.linalg.norm(coordsL-coordsR, axis=1)
        if np.abs(np.max(dists) - np.min(dists)) > TOL:
            raise ValueError 


def make_periodic_translational(mesh, x1=None, x2=None, y1=None, y2=None):
    '''
    This function imposes translational periodicity on the mesh.

    Inputs:
    -------
        mesh: mesh object
        x1: name of 1st periodic boundary in x-direction
        x2: name of 2nd periodic boundary in x-direction
        y1: name of 1st periodic boundary in y-direction
        y2: name of 2nd periodic boundary in y-direction

    Outputs:
    --------
        mesh: mesh object (modified)
    '''
    ''' Reorder nodes '''
    old_to_new_node_map = np.zeros(mesh.num_nodes, dtype=int) - 1  
        # old_to_new_node_map[n] = the new node ID of the nth node
        # (pre-ordering)
    new_to_old_node_map = np.zeros(mesh.num_nodes, dtype=int) - 1
        # new_to_old_node_map[i] = the old node ID of the ith node 
        # (post-reordering)
    next_node_id = 0

    # x
    boundary_group_x1, boundary_group_x2, node_pairs_x, next_node_id = \
            reorder_periodic_boundary_nodes(mesh, x1, x2, 0, 
            old_to_new_node_map, new_to_old_node_map, next_node_id)
    # y
    boundary_group_y1, boundary_group_y2, node_pairs_y, next_node_id = \
            reorder_periodic_boundary_nodes(mesh, y1, y2, 1,
            old_to_new_node_map, new_to_old_node_map, next_node_id)

    ''' Apply node remapping '''
    remap_nodes(mesh, old_to_new_node_map, new_to_old_node_map, next_node_id)

    ''' Match pairs of periodic boundary faces '''
    # x
    match_boundary_pair(mesh, 0, boundary_group_x1, boundary_group_x2, node_pairs_x, old_to_new_node_map, new_to_old_node_map)
    # y
    match_boundary_pair(mesh, 1, boundary_group_y1, boundary_group_y2, node_pairs_y, old_to_new_node_map, new_to_old_node_map)

    ''' Update boundary group numbers '''
    update_boundary_group_nums(mesh)

    ''' Verify valid mesh '''
    verify_periodic_mesh(mesh)

    ''' Update elements '''
    mesh.create_elements()