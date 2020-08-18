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
        icoord: which spatial dimension to check (0 for x, 1 for y)
    
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


def match_boundary_pair(mesh, icoord, boundary_group1, boundary_group2, 
        node_pairs, idx_in_node_pairs, old_to_new_node_map, 
        new_to_old_node_map):
    '''
    NOTE: only q = 1 nodes are matched
    '''
    gbasis = mesh.gbasis

    if boundary_group1 is None and boundary_group2 is None:
        return
    elif boundary_group1 is None or boundary_group2 is None:
        raise ValueError("Only one boundary group provided")

    new_to_newer_node_map = np.arange(mesh.num_nodes)
    # NodesChanged = False

    # if NodePairsA is not None:
    #     NodePairsB = NodePairsA.copy()
    #     if idx_in_node_pairsA is None:
    #         raise ValueError
    #     else:
    #         idx_in_node_pairsB = idx_in_node_pairsA.copy()

    interior_faces = mesh.interior_faces
    # icoord = icoord

    '''
    Remap node_pairs and idx_in_node_pairs
    '''
    new_node_pairs = np.zeros_like(node_pairs, dtype=int) - 1
    idx_in_node_pairs[:] = -1
    # for i in range(len(node_pairs)):
    #     new_node_pairs[i,:] = old_to_new_node_map[node_pairs[i,:]]
    new_node_pairs = old_to_new_node_map[node_pairs]
    # for i in range(len(node_pairs)):
    #     n1 = new_node_pairs[i,0]
    #     idx_in_node_pairs[n1,0] = i; idx_in_node_pairs[n1,1] = 0
    #     n2 = new_node_pairs[i,1]
    #     idx_in_node_pairs[n2,0] = i; idx_in_node_pairs[n2,1] = 1
    n1 = new_node_pairs[:,0]
    idx_in_node_pairs[n1,0] = np.arange(n1.shape[0])
    idx_in_node_pairs[n1,1] = 0
    n2 = new_node_pairs[:,1]
    idx_in_node_pairs[n2,0] = np.arange(n2.shape[0])
    idx_in_node_pairs[n2,1] = 1

    node_pairs = new_node_pairs

    # sanity check
    if np.amin(new_node_pairs) == -1:
        raise ValueError


    ''' 
    Identify and create periodic interior_faces 
    '''
    for boundary_face1 in boundary_group1.boundary_faces:
        # Extract info
        elem_id1 = boundary_face1.elem_id
        face1 = boundary_face1.face_id

        # Local IDs of face nodes
        local_node_ids = gbasis.get_local_face_principal_node_nums(
            mesh.gorder, face1)
        # Global IDs of face nodes
        global_node_ids = mesh.elem_to_node_ids[elem_id1][local_node_ids]
        # Sort for easy comparison later
        global_node_ids_1 = np.sort(global_node_ids)

        # Physical coordinates of global nodes
        # coords1 = mesh.node_coords[gfnodes1]

        # Pair each node with corresponding one on other boundary
        for boundary_face2 in boundary_group2.boundary_faces:
            # Extract info
            elem_id2 = boundary_face2.elem_id
            face2 = boundary_face2.face_id

            # Local IDs of face nodes
            local_node_ids = gbasis.get_local_face_principal_node_nums(
                mesh.gorder, face2)
            # Global IDs of face nodes
            global_node_ids = mesh.elem_to_node_ids[elem_id2][local_node_ids]

            # Physical coordinates of global nodes
            # coords2 = mesh.node_coords[gfnodes2]

            ''' Check for complete match between all nodes '''
            global_node_ids_2 = np.sort(global_node_ids)
            # Find nodes on boundary 2 paired with those in nodesort1
            idx1 = idx_in_node_pairs[global_node_ids_1, 0]
            nodepairs2 = node_pairs[idx1,1]
            nodepairssort2 = np.sort(node_pairs[idx1,1])
            match = False
            # if which_dim == 2: code.interact(local=locals())
            if np.all(global_node_ids_2 == nodepairssort2):
                # this means the nodes match, but the order may not necessarily be the same
                match = True


            # match = False
            # for n in range(nfnode):
            #     coord1 = coords1[n,:]
            #     for m in range(nfnode):
            #         coord2 = coords2[m,:]
            #         # Find distance between the two nodes
            #         norm = np.linalg.norm(coord1-coord2, ord=1)
            #         # Check if distance is equal to pdiff
            #         if np.abs(norm-pdiff) < TOL:
            #             match = True
            #             # Force nodes to match exactly
            #             for d in range(mesh.dim):
            #                 if d == icoord: continue # skip periodic direction
            #                 # code.interact(local=locals())
            #                 coord2[d] = coord1[d]
            #                 mesh.node_coords[gfnodes2[m],d] = coord1[d]
            #             break
            #         else:
            #             match = False

            #     if not match: 
            #         # could not match this node
            #         break

            # # If match is true, then we have matching faces
            # if match:
                if not np.all(global_node_ids_2 == nodepairs2):

                    if icoord == 0:
                        raise Exception

                    # node order doesn't match, so reorder
                    mesh.node_coords[nodepairssort2] = mesh.node_coords[nodepairs2]

                    # store for remapping elements to nodes later
                    new_to_newer_node_map[nodepairs2] = nodepairssort2

                    # Modify pairing as necessary
                    node_pairs[idx1,1] = nodepairssort2
                    idx_in_node_pairs[nodepairssort2,0] = idx1

                    # Modify other pairings
                    # if NodePairsA is not None:
                    #     for n in range(nfnode):
                    #         # check if node has been swapped
                    #         n2 = nodepairs2[n]
                    #         ns2 = nodepairssort2[n]
                    #         if n2 != ns2:
                    #             # check if it exists in other pairing
                    #             idxA = idx_in_node_pairsA[n2,0]
                    #             if idxA != -1:
                    #                 b = idx_in_node_pairsA[n2,1]
                    #                 NodePairsA[idxA, b] = ns2
                    #                 # reset n2
                    #                 idx_in_node_pairsB[n2, :] = -1
                    #                 idx_in_node_pairsB[ns2, 0] = idxA
                    #                 idx_in_node_pairsB[ns2, 1] = b

                    #     # Put back into A
                    #     idx_in_node_pairsA[:] = idx_in_node_pairsB[:]


                    # NodesChanged = True
                    # remap elements
                    for elem in range(mesh.num_elems):
                        mesh.elem_to_node_ids[elem, :] = new_to_newer_node_map[mesh.elem_to_node_ids[elem, :]]
                    # reset new_to_newer_node_map
                    new_to_newer_node_map = np.arange(mesh.num_nodes)

                    
                # Create IFace between these two faces
                mesh.num_interior_faces += 1
                interior_faces.append(mesh_defs.InteriorFace())
                interior_face = interior_faces[-1]
                interior_face.elemL_id = elem_id1
                interior_face.faceL_id = face1
                interior_face.elemR_id = elem_id2
                interior_face.faceR_id = face2

                boundary_group1.num_boundary_faces -= 1
                boundary_group2.num_boundary_faces -= 1
                break




        if not match:
            raise ValueError("Could not find matching boundary face")


    # Verification
    if boundary_group1.num_boundary_faces != 0 or boundary_group2.num_boundary_faces != 0:
        raise ValueError
    mesh.num_boundary_groups -= 2


    # if NodesChanged:
    #     # remap elements
    #     mesh = mesh.ElemGroups[0]
    #     num_elems = mesh.num_elems
    #     for elem in range(num_elems):
    #         mesh.elem_to_node_ids[elem, :] = new_to_newer_node_map[mesh.elem_to_node_ids[elem, :]]

    # mesh.boundary_groups.remove(boundary_group1)
    # mesh.boundary_groups.remove(boundary_group2)
    mesh.boundary_groups.pop(boundary_group1.name)
    mesh.boundary_groups.pop(boundary_group2.name)

    # print
    if icoord == 0:
        s = "x"
    elif icoord == 1:
        s = "y"
    else:
        s = "z"
    print("Matched periodic boundaries in %s" % (s))


def reorder_periodic_boundary_nodes(mesh, b1, b2, icoord, 
        old_to_new_node_map, new_to_old_node_map, next_node_id):

    gbasis = mesh.gbasis

    if b1 is None and b2 is None:
        return None, None, None, None, next_node_id
    elif b1 == b2:
        raise ValueError("Duplicate boundaries")

    node_pairs = np.zeros([mesh.num_nodes, 2], dtype=int) - 1
    idx_in_node_pairs = np.zeros([mesh.num_nodes, 2], dtype=int) - 1
    num_node_pairs = 0
    # node2_matched = [False]*mesh.num_nodes
    node2_matched = np.zeros(mesh.num_nodes, dtype=bool)

    # Extract the two boundary_groups
    boundary_group1 = mesh.boundary_groups[b1]
    boundary_group2 = mesh.boundary_groups[b2]

    start_node_id = next_node_id

    # Sanity check
    # if boundary_group1 is None or boundary_group2 is None:
    #     raise Exception("One or both boundaries not found")
    # elif boundary_group1 == boundary_group2:
    #     raise Exception("Duplicate boundaries")

    # icoord = icoord
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
    # distance between the two boundaries
    pdiff = np.abs(pcoord1-pcoord2)

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
                # has not been ordered yet
                old_to_new_node_map[node_id] = next_node_id
                new_to_old_node_map[next_node_id] = node_id
                next_node_id += 1
            if idx_in_node_pairs[node_id, 0] == -1:
                node_pairs[num_node_pairs, 0] = node_id
                idx_in_node_pairs[node_id, 0] = num_node_pairs
                idx_in_node_pairs[node_id, 1] = 0
                num_node_pairs += 1

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
            ''' Find match with boundary group 1 '''
            # Unless already ordered 
            # if old_to_new_node_map[node_id] == -1: # has not been ordered yet

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
                    idx1 = idx_in_node_pairs[node1_id, 0]
                    node_pairs[idx1, 1] = node2_id
                    idx_in_node_pairs[node2_id, 0] = idx1
                    idx_in_node_pairs[node2_id, 1] = 1

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
    node_pairs = node_pairs[:num_node_pairs,:]

    # Print info
    if icoord == 0:
        s = "x"
    elif icoord == 1:
        s = "y"
    else:
        s = "z"
    print("Reordered periodic boundary in %s-direction" % (s))

    return boundary_group1, boundary_group2, node_pairs, idx_in_node_pairs, \
            next_node_id


def remap_nodes(mesh, old_to_new_node_map, new_to_old_node_map, 
        next_node_id=-1):
    # nPeriodicNode = next_node_id
    # NewCoords = np.zeros_like(mesh.node_coords)

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
    # Newelem_to_node_ids = np.zeros_like(mesh.elem_to_node_ids, dtype=int) - 1
    num_elems = mesh.num_elems
    for elem_id in range(num_elems):
        mesh.elem_to_node_ids[elem_id,:] = old_to_new_node_map[
                mesh.elem_to_node_ids[elem_id, :]]

    # Store in mesh
    # mesh.node_coords = NewCoords
    # mesh.elem_to_node_ids = Newelem_to_node_ids


def verify_periodic_mesh(mesh):
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


def update_boundary_group_nums(mesh):
    i = 0
    for boundary_group in mesh.boundary_groups.values():
        boundary_group.number = i
        i += 1


def make_periodic_translational(mesh, x1=None, x2=None, y1=None, y2=None):

    ''' Reorder nodes '''
    old_to_new_node_map = np.zeros(mesh.num_nodes, dtype=int) - 1  
        # old_to_new_node_map[n] = the new node number (post-reordering) of node n (pre-ordering)
    new_to_old_node_map = np.zeros(mesh.num_nodes, dtype=int) - 1
        # new_to_old_node_map[i] = the node number (pre-reordering) of the ith node (post-reordering)
    next_node_id = 0

    # x
    boundary_group_x1, boundary_group_x2, node_pairs_x, \
            idx_in_node_pairs_x, next_node_id = \
            reorder_periodic_boundary_nodes(mesh, x1, x2, 0, 
            old_to_new_node_map, new_to_old_node_map, next_node_id)
    # y
    boundary_group_y1, boundary_group_y2, node_pairs_y, \
            idx_in_node_pairs_y, next_node_id = \
            reorder_periodic_boundary_nodes(mesh, y1, y2, 1, 
            old_to_new_node_map, new_to_old_node_map, next_node_id)
    # z
    # boundary_groupZ1, boundary_groupZ2, NodePairsZ, idx_in_node_pairsZ, next_node_id = reorder_periodic_boundary_nodes(mesh, 
    #     z1, z2, 2, old_to_new_node_map, new_to_old_node_map, next_node_id)


    ''' Remap nodes '''
    remap_nodes(mesh, old_to_new_node_map, new_to_old_node_map, next_node_id)


    ''' Match pairs of periodic boundary faces '''
    # x
    match_boundary_pair(mesh, 0, boundary_group_x1, boundary_group_x2, node_pairs_x, idx_in_node_pairs_x, old_to_new_node_map, new_to_old_node_map)
    # y
    match_boundary_pair(mesh, 1, boundary_group_y1, boundary_group_y2, node_pairs_y, idx_in_node_pairs_y, old_to_new_node_map, new_to_old_node_map) #, 
        # NodePairsZ, idx_in_node_pairsZ)
    # z
    # match_boundary_pair(mesh, 2, boundary_groupZ1, boundary_groupZ2, NodePairsZ, idx_in_node_pairsZ, old_to_new_node_map, new_to_old_node_map)


    ''' Update face orientations '''
    # making it periodic messes things up
    # check_face_orientations(mesh)

    ''' Update boundary group numbers '''
    update_boundary_group_nums(mesh)

    ''' Verify valid mesh '''
    verify_periodic_mesh(mesh)

    ''' Update elements '''
    mesh.create_elements()