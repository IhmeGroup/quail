import code
import numpy as np 

import data

import meshing.meshbase as mesh_defs

import numerics.basis.tools as basis_tools

tol = 1.e-10

def element_volumes(mesh, solver=None):
    '''
    Method: element_volumes
    --------------------------
    Calculates total and per element volumes

    INPUTS:
        mesh: mesh object
        solver: type of solver (i.e. DG, ADER-DG, etc...)
    
    OUTPUTS:
        TotalVolume: total volume in the mesh
        ElemVolumes: volume at each element
    '''
    # Check if already calculated
    if solver is not None:
        if hasattr(solver.DataSet, "TotalVolume") \
            and hasattr(solver.DataSet, "ElemVolumes"):
                return solver.DataSet.TotalVolume, solver.DataSet.ElemVolumes

    ElemVolumes = np.zeros(mesh.num_elems)
    TotalVolume = 0.

    gorder = mesh.gorder
    gbasis = mesh.gbasis

    quad_order = gbasis.get_quadrature_order(mesh,gorder)
    xq, wq = gbasis.get_quadrature_data(quad_order)

    # xq = gbasis.quad_pts
    # wq = gbasis.quad_wts
    nq = xq.shape[0]

    for elem in range(mesh.num_elems):
        djac,_,_ = basis_tools.element_jacobian(mesh,elem,xq,get_djac=True)

        # for iq in range(nq):
        #     ElemVolumes[elem] += wq[iq] * JData.djac[iq*(JData.nq != 1)]
        ElemVolumes[elem] = np.sum(wq*djac)

        TotalVolume += ElemVolumes[elem]

    if solver is not None:
        solver.DataSet.TotalVolume = TotalVolume
        solver.DataSet.ElemVolumes = ElemVolumes

    return TotalVolume, ElemVolumes


def get_element_centroid(mesh, elem):
    gbasis = mesh.gbasis
    xcentroid = mesh_defs.ref_to_phys(mesh, elem, mesh.gbasis.CENTROID)  

    return xcentroid


def neighbor_across_face(mesh, elem, face):
    '''
    Method: neighbor_across_face
    ------------------------------
    Identifies neighbor elements across each face

    INPUTS:
        mesh: mesh object
        elem: element index
        face: face index w.r.t. the element in ref space
    
    OUTPUTS:
        eN: element index of the neighboring face
        faceN: face index w.r.t. the neighboring element in ref space
    '''
    Face = mesh.Faces[elem][face]
    # code.interact(local=locals())
    if Face.Type == mesh_defs.FaceType.Interior:
        iiface = Face.Number
        eN  = mesh.interior_faces[iiface].elemR_id
        faceN = mesh.interior_faces[iiface].faceR_id

        if eN == elem:
            eN  = mesh.interior_faces[iiface].elemL_id
            faceN = mesh.interior_faces[iiface].faceL_id
    else:
        eN    = -1
        faceN = -1

    return eN, faceN


def check_face_orientations(mesh):
    '''
    Method: check_face_orientations
    --------------------------------
    Checks the face orientations for 2D meshes

    INPUTS:
        mesh: mesh object
    
    NOTES:
        only returns a message if an error exists
    '''
    gbasis = mesh.gbasis
    if mesh.dim == 1:
        # don't need to check for 1D
        return

    for IFace in mesh.interior_faces:
        elemL_id = IFace.elemL_id
        elemR_id = IFace.elemR_id
        faceL_id = IFace.faceL_id
        faceR_id = IFace.faceR_id

        elemL_nodes = mesh.elements[elemL_id].node_ids
        elemR_nodes = mesh.elements[elemR_id].node_ids

        # Get local q=1 nodes on face for left element
        lfnodes = gbasis.get_local_face_principal_node_nums(mesh.gorder, faceL_id)
        # Convert to global node numbering
        # gfnodesL = mesh.elem_to_node_ids[elemL][lfnodes]
        gfnodesL = elemL_nodes[lfnodes]

        # Get local q=1 nodes on face for right element
        lfnodes = gbasis.get_local_face_principal_node_nums(mesh.gorder, faceR_id)
        # Convert to global node numbering
        # gfnodesR = mesh.elem_to_node_ids[elemR][lfnodes]
        gfnodesR = elemR_nodes[lfnodes]

        # Node Ordering should be reversed between the two elements
        if not np.all(gfnodesL == gfnodesR[::-1]):
            raise Exception("Face orientation for elemL_id = %d, elemR_id = %d \\ is incorrect"
                % (elemL_id, elemR_id))


def RandomizeNodes(mesh, elem = -1, orient = -1):
    OldNode2NewNode = np.arange(mesh.num_nodes)
    np.random.shuffle(OldNode2NewNode)
    NewNodeOrder = np.zeros(mesh.num_nodes, dtype=int) - 1
    NewNodeOrder[OldNode2NewNode] = np.arange(mesh.num_nodes)

    if np.min(NewNodeOrder) == -1:
        raise ValueError

    RemapNodes(mesh, OldNode2NewNode, NewNodeOrder)

    # for i in range(mesh.num_nodes):
    #   n = OldNode2NewNode[i]
    #     NewNodeOrder[n] = i
    #         # NewNodeOrder[i] = the node number (pre-reordering) of the ith node (post-reordering)

    print("Randomized nodes")


def RotateNodes(mesh, theta_x = 0., theta_y = 0., theta_z = 0.):

    # Construct rotation matrices
    if mesh.dim == 3:
        Rx = np.array([[1., 0., 0.], [0., np.cos(theta_x), -np.sin(theta_x)], [0., np.sin(theta_x), np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y), 0., np.sin(theta_y)], [0., 1., 0.], [-np.sin(theta_y), 0., np.cos(theta_y)]])
        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0.], [np.sin(theta_z), np.cos(theta_z), 0.], [0., 0., 1.]])
        R = np.matmul(Rx, np.matmul(Ry, Rz))
    elif mesh.dim == 2:
        R = np.array([[np.cos(theta_z), -np.sin(theta_z)], [np.sin(theta_z), np.cos(theta_z)]])
    else:
        raise NotImplementedError

    mesh.node_coords = np.matmul(R, mesh.node_coords.transpose()).transpose()

    print("Rotated nodes")


def VerifyPeriodicBoundary(mesh, BFG, icoord):
    coord = np.nan
    gbasis = mesh.gbasis
    for BF in BFG.BFaces:
        # Extract info
        elem_id = BF.elem_id
        face = BF.face_id

        ''' Get physical coordinates of face '''
        # Get local q = 1 nodes on face
        lfnodes = gbasis.get_local_face_principal_node_nums(mesh.gorder, face)

        # Convert to global node numbering
        # gfnodes = mesh.elem_to_node_ids[elem_id][lfnodes]

        # Physical coordinates of global nodes
        # coords = mesh.node_coords[gfnodes]
        elem_coords = mesh.elements[elem_id].node_coords
        coords = elem_coords[lfnodes]
        if np.isnan(coord):
            coord = coords[0, icoord]

        # Make sure coord1 matches for all nodes
        if np.any(np.abs(coords[:,icoord] - coord) > tol):
            raise ValueError

        # Now force each node to have the same exact coord1 
        coords[:,icoord] = coord

    return coord


def MatchBoundaryPair(mesh, which_dim, BFG1, BFG2, NodePairs, idx_in_node_pairs, OldNode2NewNode, NewNodeOrder,
    NodePairsA = None, idx_in_node_pairsA = None):
    '''
    NOTE: only q = 1 nodes are matched
    '''
    gbasis = mesh.gbasis

    if BFG1 is None and BFG2 is None:
        return

    NewNode2NewerNode = np.arange(mesh.num_nodes)
    NodesChanged = False

    if NodePairsA is not None:
        NodePairsB = NodePairsA.copy()
        if idx_in_node_pairsA is None:
            raise ValueError
        else:
            idx_in_node_pairsB = idx_in_node_pairsA.copy()

    interior_faces = mesh.interior_faces
    icoord = which_dim

    # Extract relevant BFGs
    # BFG1 = None; BFG2 = None;
    # for BFG in mesh.boundary_groups:
    #     if BFG.Name == b1:
    #         BFG1 = BFG
    #     if BFG.Name == b2:
    #         BFG2 = BFG
    # BFG = None

    # # Sanity check
    # if BFG1 is None or BFG2 is None:
    #     raise Exception("One or both boundaries not found")
    # elif BFG1 == BFG2:
    #     raise Exception("Duplicate boundaries")

    # icoord = which_dim
    # if icoord < 0 or icoord >= mesh.dim:
    #     raise ValueError

    # ''' 
    # Make sure each boundary is compatible with periodicity 
    # Note: the boundary node coordinates may be slightly modified
    # to ensure same coordinate in periodic direction
    # '''
    # if BFG1.nBFace != BFG2.nBFace:
    #     raise ValueError

    # pcoord1 = VerifyPeriodicBoundary(mesh, BFG1, icoord)
    # pcoord2 = VerifyPeriodicBoundary(mesh, BFG2, icoord)
    # # distance between the two boundaries
    # pdiff = np.abs(pcoord1-pcoord2)


    '''
    Remap NodePairs and idx_in_node_pairs
    '''
    NewNodePairs = np.zeros_like(NodePairs, dtype=int) - 1
    idx_in_node_pairs[:] = -1
    for i in range(len(NodePairs)):
        NewNodePairs[i,:] = OldNode2NewNode[NodePairs[i,:]]
    for i in range(len(NodePairs)):
        n1 = NewNodePairs[i,0]
        idx_in_node_pairs[n1,0] = i; idx_in_node_pairs[n1,1] = 0
        n2 = NewNodePairs[i,1]
        idx_in_node_pairs[n2,0] = i; idx_in_node_pairs[n2,1] = 1

    NodePairs = NewNodePairs

    # sanity check
    if np.amin(NewNodePairs) == -1:
        raise ValueError


    ''' 
    Identify and create periodic interior_faces 
    '''
    for BFace1 in BFG1.BFaces:
        # Extract info
        elem_id1 = BFace1.elem_id
        face1 = BFace1.face_id

        ''' Get physical coordinates of face '''
        # Get local q = 1 nodes on face
        lfnodes1 = gbasis.get_local_face_principal_node_nums( 
            mesh.gorder, face1)
        nfnode = lfnodes1.shape[0]

        # Convert to global node numbering
        gfnodes1 = mesh.elem_to_node_ids[elem_id1][lfnodes1]
        nodesort1 = np.sort(gfnodes1)

        # Physical coordinates of global nodes
        coords1 = mesh.node_coords[gfnodes1]

        # Pair each node with corresponding one on other boundary
        for BFace2 in BFG2.BFaces:
            # Extract info
            elem_id2 = BFace2.elem_id
            face2 = BFace2.face_id

            ''' Get physical coordinates of face '''
            # Get local q = 1 nodes on face
            lfnodes2 = gbasis.get_local_face_principal_node_nums(
                mesh.gorder, face2)

            # Convert to global node numbering
            gfnodes2 = mesh.elem_to_node_ids[elem_id2][lfnodes2]

            # Physical coordinates of global nodes
            coords2 = mesh.node_coords[gfnodes2]

            ''' Check for complete match between all nodes '''
            nodesort2 = np.sort(gfnodes2)
            # Find nodes on boundary 2 paired with those in nodesort1
            idx1 = idx_in_node_pairs[nodesort1, 0]
            nodepairs2 = NodePairs[idx1,1]
            nodepairssort2 = np.sort(NodePairs[idx1,1])
            match = False
            # if which_dim == 2: code.interact(local=locals())
            if np.all(nodesort2 == nodepairssort2):
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
            #         if np.abs(norm-pdiff) < tol:
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
                if not np.all(nodesort2 == nodepairs2):

                    if which_dim == 0:
                        raise Exception

                    # node order doesn't match, so reorder
                    mesh.node_coords[nodepairssort2] = mesh.node_coords[nodepairs2]

                    # store for remapping elements to nodes later
                    NewNode2NewerNode[nodepairs2] = nodepairssort2

                    # Modify pairing as necessary
                    NodePairs[idx1,1] = nodepairssort2
                    idx_in_node_pairs[nodepairssort2,0] = idx1

                    # Modify other pairings
                    if NodePairsA is not None:
                        for n in range(nfnode):
                            # check if node has been swapped
                            n2 = nodepairs2[n]
                            ns2 = nodepairssort2[n]
                            if n2 != ns2:
                                # check if it exists in other pairing
                                idxA = idx_in_node_pairsA[n2,0]
                                if idxA != -1:
                                    b = idx_in_node_pairsA[n2,1]
                                    NodePairsA[idxA, b] = ns2
                                    # reset n2
                                    idx_in_node_pairsB[n2, :] = -1
                                    idx_in_node_pairsB[ns2, 0] = idxA
                                    idx_in_node_pairsB[ns2, 1] = b

                        # Put back into A
                        idx_in_node_pairsA[:] = idx_in_node_pairsB[:]


                    NodesChanged = True
                    # remap elements
                    for elem in range(mesh.num_elems):
                        mesh.elem_to_node_ids[elem, :] = NewNode2NewerNode[mesh.elem_to_node_ids[elem, :]]
                    # reset NewNode2NewerNode
                    NewNode2NewerNode = np.arange(mesh.num_nodes)

                    
                # Create IFace between these two faces
                mesh.num_interior_faces += 1
                interior_faces.append(mesh_defs.InteriorFace())
                IF = interior_faces[-1]
                IF.elemL_id = elem_id1
                IF.faceL_id = face1
                IF.elemR_id = elem_id2
                IF.faceR_id = face2

                BFG1.nBFace -= 1
                BFG2.nBFace -= 1
                break




        if not match:
            raise Exception("Could not find matching boundary face")


    # Verification
    if BFG1.nBFace != 0 or BFG2.nBFace != 0:
        raise ValueError
    mesh.num_boundary_groups -= 2


    # if NodesChanged:
    #     # remap elements
    #     mesh = mesh.ElemGroups[0]
    #     num_elems = mesh.num_elems
    #     for elem in range(num_elems):
    #         mesh.elem_to_node_ids[elem, :] = NewNode2NewerNode[mesh.elem_to_node_ids[elem, :]]

    # mesh.boundary_groups.remove(BFG1)
    # mesh.boundary_groups.remove(BFG2)
    mesh.boundary_groups.pop(BFG1.name)
    mesh.boundary_groups.pop(BFG2.name)

    # print
    if which_dim == 0:
        s = "x"
    elif which_dim == 1:
        s = "y"
    else:
        s = "z"
    print("Matched periodic boundaries in %s" % (s))


def ReorderPeriodicBoundaryNodes(mesh, b1, b2, which_dim, OldNode2NewNode, NewNodeOrder, NextIdx):

    gbasis = mesh.gbasis

    if b1 is None and b2 is None:
        return None, None, None, None, NextIdx
    elif b1 == b2:
        raise ValueError("Duplicate boundaries")

    NodePairs = np.zeros([mesh.num_nodes, 2], dtype=int) - 1
    idx_in_node_pairs = np.zeros([mesh.num_nodes, 2], dtype=int) - 1
    num_node_pairs = 0
    node2_matched = [False]*mesh.num_nodes

    # Extract relevant BFGs
    # BFG1 = None; BFG2 = None;
    # for BFG in mesh.boundary_groups:
    #     if BFG.Name == b1:
    #         BFG1 = BFG
    #     if BFG.Name == b2:
    #         BFG2 = BFG
    # BFG = None
    BFG1 = mesh.boundary_groups[b1]
    BFG2 = mesh.boundary_groups[b2]

    StartIdx = NextIdx

    # Sanity check
    # if BFG1 is None or BFG2 is None:
    #     raise Exception("One or both boundaries not found")
    # elif BFG1 == BFG2:
    #     raise Exception("Duplicate boundaries")

    icoord = which_dim
    if icoord < 0 or icoord >= mesh.dim:
        raise ValueError

    ''' 
    Make sure each boundary is compatible with periodicity 
    Note: the boundary node coordinates may be slightly modified
    to ensure same coordinate in periodic direction
    '''
    if BFG1.nBFace != BFG2.nBFace:
        raise ValueError

    pcoord1 = VerifyPeriodicBoundary(mesh, BFG1, icoord)
    pcoord2 = VerifyPeriodicBoundary(mesh, BFG2, icoord)
    # distance between the two boundaries
    pdiff = np.abs(pcoord1-pcoord2)


    '''
    Deal with first boundary
    '''

    # Get b1 nodes
    for BFace in BFG1.BFaces:
        # Extract info
        elem = BFace.elem_id
        face = BFace.face_id

        ''' Get physical coordinates of face '''
        # Get local q = 1 nodes on face
        lfnodes = gbasis.get_local_face_principal_node_nums( 
                mesh.gorder, face)

        # Convert to global node numbering
        gfnodes = mesh.elem_to_node_ids[elem][lfnodes[:]]

        for node in gfnodes:
            if OldNode2NewNode[node] == -1: # has not been ordered yet
                OldNode2NewNode[node] = NextIdx
                NewNodeOrder[NextIdx] = node
                NextIdx += 1
            if idx_in_node_pairs[node,0] == -1:
                NodePairs[num_node_pairs,0] = node
                idx_in_node_pairs[node,0] = num_node_pairs
                idx_in_node_pairs[node,1] = 0
                num_node_pairs += 1

    StopIdx = NextIdx

    '''
    Deal with second boundary
    '''

    for BFace in BFG2.BFaces:
        # Extract info
        elem = BFace.elem_id
        face = BFace.face_id

        ''' Get physical coordinates of face '''
        # Get local q = 1 nodes on face
        lfnodes = gbasis.get_local_face_principal_node_nums( 
                mesh.gorder, face)

        # Convert to global node numbering
        gfnodes = mesh.elem_to_node_ids[elem][lfnodes[:]]

        for node2 in gfnodes:
            ''' Find match with boundary 1 '''
            # Unless already ordered 
            # if OldNode2NewNode[node2] == -1: # has not been ordered yet

            if node2_matched[node2]: 
                # this node already matched - skip

                # sanity check
                if OldNode2NewNode[node2] == -1:
                    raise Exception

                continue

            coord2 = mesh.node_coords[node2]
            
            match = False
            for n in range(num_node_pairs):
                if NodePairs[n,1] != -1:
                    # node1 already paired - skip
                    continue
                node1 = NodePairs[n,0]

                coord1 = mesh.node_coords[node1]
                # Find distance between the two nodes
                norm = np.linalg.norm(coord1-coord2, ord=1)
                # Check if distance is equal to pdiff
                if np.abs(norm-pdiff) < tol:
                    match = True
                    if OldNode2NewNode[node2] == -1:
                        # node2 not reordered yet
                        # Now order node2 corresponding to node1
                        idx2 = OldNode2NewNode[node1] + StopIdx - StartIdx
                        OldNode2NewNode[node2] = idx2
                        NewNodeOrder[idx2] = node2
                        NextIdx = np.amax([NextIdx, idx2])
                        # Force nodes to match exactly
                        for d in range(mesh.dim):
                            if d == icoord: continue # skip periodic direction
                            # code.interact(local=locals())
                            coord2[d] = coord1[d]
                    # Store node pair
                    idx1 = idx_in_node_pairs[node1, 0]
                    NodePairs[idx1, 1] = node2
                    idx_in_node_pairs[node2, 0] = idx1
                    idx_in_node_pairs[node2, 1] = 1
                    # Flag node2 as matched
                    node2_matched[node2] = True
                    break

            # # Loop through the newly ordered periodic nodes on boundary 1
            # match = False
            # for idx in range(StopIdx-StartIdx):
            #     node1 = NewNodeOrder[StartIdx+idx]
            #     # If node1 already matched, skip
            #     if idx_in_node_pairs[node1] != -1:
            #         continue
            #     coord1 = mesh.node_coords[node1]
            #     # Find distance between the two nodes
            #     norm = np.linalg.norm(coord1-coord2, ord=1)
            #     # Check if distance is equal to pdiff
            #     if np.abs(norm-pdiff) < tol:
            #         match = True
            #         if OldNode2NewNode[node2] == -1:
            #             # Now order node2 corresponding to node1
            #             idx2 = StopIdx + idx
            #             OldNode2NewNode[node2] = idx2
            #             NewNodeOrder[idx2] = node2
            #             NextIdx = np.amax([NextIdx, idx2])
            #             # Force nodes to match exactly
            #             for d in range(mesh.dim):
            #                 if d == icoord: continue # skip periodic direction
            #                 # code.interact(local=locals())
            #                 coord2[d] = coord1[d]
            #         # Store node pair
            #         NodePairs[nNodePair] = np.array([node1, node2])
            #         idx_in_node_pairs[node1] = nNodePair
            #         nNodePair += 1
            #         break

            if not match:
                print("node2 = %d"  % (node2))
                print(coord2)
                raise Exception("Could not find matching boundary face")

    # sanity check
    if StartIdx != StopIdx:
        # This means at least one pair of nodes was matched
        NextIdx += 1
    if NextIdx != 2*StopIdx - StartIdx:
        code.interact(local=locals())
        raise ValueError

    # resize NodePairs
    NodePairs = NodePairs[:num_node_pairs,:]

    # print
    if which_dim == 0:
        s = "x"
    elif which_dim == 1:
        s = "y"
    else:
        s = "z"
    print("Reordered periodic boundary in %s" % (s))

    return BFG1, BFG2, NodePairs, idx_in_node_pairs, NextIdx


def RemapNodes(mesh, OldNode2NewNode, NewNodeOrder, NextIdx=-1):
    # nPeriodicNode = NextIdx
    # NewCoords = np.zeros_like(mesh.node_coords)

    # Fill up OldNode2NewNode and NewNodeOrder with non-periodic nodes
    if NextIdx != -1:
        for node in range(mesh.num_nodes):
            if OldNode2NewNode[node] == -1: # has not been ordered yet
                OldNode2NewNode[node] = NextIdx
                NewNodeOrder[NextIdx] = node
                NextIdx += 1

    # New coordinate list
    # NewCoords = mesh.node_coords[NewNodeOrder]
    mesh.node_coords[:] = mesh.node_coords[NewNodeOrder]

    # New elem_to_node_ids
    Newelem_to_node_ids = np.zeros_like(mesh.elem_to_node_ids, dtype=int) - 1
    num_elems = mesh.num_elems
    for elem in range(num_elems):
        mesh.elem_to_node_ids[elem,:] = OldNode2NewNode[mesh.elem_to_node_ids[elem, :]]

    # Store in mesh
    # mesh.node_coords = NewCoords
    # mesh.elem_to_node_ids = Newelem_to_node_ids


def VerifyPeriodicMesh(mesh):
    # Loop through interior faces
    for IF in mesh.interior_faces:
        elemL = IF.elemL_id
        elemR = IF.elemR_id
        faceL_id = IF.faceL_id
        faceR_id = IF.faceR_id
        gbasis = mesh.gbasis
        gorder = mesh.gorder

        ''' Get global face nodes - left '''
        fnodesL = gbasis.get_local_face_principal_node_nums( 
            gorder, faceL_id)

        # Convert to global node numbering
        fnodesL = mesh.elem_to_node_ids[elemL][fnodesL[:]]

        ''' Get global face nodes - right '''
        fnodesR = gbasis.get_local_face_principal_node_nums( 
            gorder, faceR_id)

        # Convert to global node numbering
        fnodesR = mesh.elem_to_node_ids[elemR][fnodesR[:]]

        ''' If exact same global nodes, then this is NOT a periodic face '''
        fnodesLsort = np.sort(fnodesL)
        fnodesRsort = np.sort(fnodesR)
        if np.all(fnodesLsort == fnodesRsort):
            # skip non-periodic face
            continue

        ''' Compare distances '''
        coordsL = mesh.node_coords[fnodesLsort]
        coordsR = mesh.node_coords[fnodesRsort]
        dists = np.linalg.norm(coordsL-coordsR, axis=1)
        if np.abs(np.max(dists) - np.min(dists)) > tol:
            raise ValueError        


def update_boundary_group_nums(mesh):
    i = 0
    for BFG in mesh.boundary_groups.values():
        BFG.number = i
        i += 1

def MakePeriodicTranslational(mesh, x1=None, x2=None, y1=None, y2=None, z1=None, z2=None):

    ''' Reorder nodes '''
    OldNode2NewNode = np.zeros(mesh.num_nodes, dtype=int)-1  
        # OldNode2NewNode[n] = the new node number (post-reordering) of node n (pre-ordering)
    NewNodeOrder = np.zeros(mesh.num_nodes, dtype=int)-1
        # NewNodeOrder[i] = the node number (pre-reordering) of the ith node (post-reordering)
    NextIdx = 0

    # x
    BFGX1, BFGX2, NodePairsX, idx_in_node_pairsX, NextIdx = ReorderPeriodicBoundaryNodes(mesh, 
        x1, x2, 0, OldNode2NewNode, NewNodeOrder, NextIdx)
    # y
    BFGY1, BFGY2, NodePairsY, idx_in_node_pairsY, NextIdx = ReorderPeriodicBoundaryNodes(mesh, 
        y1, y2, 1, OldNode2NewNode, NewNodeOrder, NextIdx)
    # z
    BFGZ1, BFGZ2, NodePairsZ, idx_in_node_pairsZ, NextIdx = ReorderPeriodicBoundaryNodes(mesh, 
        z1, z2, 2, OldNode2NewNode, NewNodeOrder, NextIdx)


    ''' Remap nodes '''
    RemapNodes(mesh, OldNode2NewNode, NewNodeOrder, NextIdx)


    ''' Match pairs of periodic boundary faces '''
    # x
    MatchBoundaryPair(mesh, 0, BFGX1, BFGX2, NodePairsX, idx_in_node_pairsX, OldNode2NewNode, NewNodeOrder)
    # y
    MatchBoundaryPair(mesh, 1, BFGY1, BFGY2, NodePairsY, idx_in_node_pairsY, OldNode2NewNode, NewNodeOrder, 
        NodePairsZ, idx_in_node_pairsZ)
    # z
    MatchBoundaryPair(mesh, 2, BFGZ1, BFGZ2, NodePairsZ, idx_in_node_pairsZ, OldNode2NewNode, NewNodeOrder)


    ''' Update face orientations '''
    # making it periodic messes things up
    # check_face_orientations(mesh)

    ''' Update boundary group numbers '''
    update_boundary_group_nums(mesh)

    ''' Verify valid mesh '''
    VerifyPeriodicMesh(mesh)

    ''' Update elements '''
    mesh.create_elements()









