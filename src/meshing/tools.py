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

    ElemVolumes = np.zeros(mesh.nElem)
    TotalVolume = 0.

    gorder = mesh.gorder
    gbasis = mesh.gbasis

    quad_order = gbasis.get_quadrature_order(mesh,gorder)
    xq, wq = gbasis.get_quadrature_data(quad_order)

    # xq = gbasis.quad_pts
    # wq = gbasis.quad_wts
    nq = xq.shape[0]

    for elem in range(mesh.nElem):
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
    xcentroid, _ = mesh_defs.ref_to_phys(mesh, elem, None, mesh.gbasis.CENTROID)  

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
        eN  = mesh.IFaces[iiface].ElemR
        faceN = mesh.IFaces[iiface].faceR

        if eN == elem:
            eN  = mesh.IFaces[iiface].ElemL
            faceN = mesh.IFaces[iiface].faceL
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
    if mesh.Dim == 1:
        # don't need to check for 1D
        return

    for IFace in mesh.IFaces:
        ielemL = IFace.ElemL
        ielemR = IFace.ElemR
        faceL = IFace.faceL
        faceR = IFace.faceR

        elemL_nodes = mesh.elements[ielemL].node_nums
        elemR_nodes = mesh.elements[ielemR].node_nums

        # Get local q=1 nodes on face for left element
        lfnodes = gbasis.get_local_face_principal_node_nums(mesh.gorder, faceL)
        # Convert to global node numbering
        # gfnodesL = mesh.Elem2Nodes[elemL][lfnodes]
        gfnodesL = elemL_nodes[lfnodes]

        # Get local q=1 nodes on face for right element
        lfnodes = gbasis.get_local_face_principal_node_nums(mesh.gorder, faceR)
        # Convert to global node numbering
        # gfnodesR = mesh.Elem2Nodes[elemR][lfnodes]
        gfnodesR = elemR_nodes[lfnodes]

        # Node Ordering should be reversed between the two elements
        if not np.all(gfnodesL == gfnodesR[::-1]):
            raise Exception("Face orientation for ielemL = %d, ielemR = %d \\ is incorrect"
                % (ielemL, ielemR))


def RandomizeNodes(mesh, elem = -1, orient = -1):
    OldNode2NewNode = np.arange(mesh.nNode)
    np.random.shuffle(OldNode2NewNode)
    NewNodeOrder = np.zeros(mesh.nNode, dtype=int) - 1
    NewNodeOrder[OldNode2NewNode] = np.arange(mesh.nNode)

    if np.min(NewNodeOrder) == -1:
        raise ValueError

    RemapNodes(mesh, OldNode2NewNode, NewNodeOrder)

    # for i in range(mesh.nNode):
    #   n = OldNode2NewNode[i]
    #     NewNodeOrder[n] = i
    #         # NewNodeOrder[i] = the node number (pre-reordering) of the ith node (post-reordering)

    print("Randomized nodes")


def RotateNodes(mesh, theta_x = 0., theta_y = 0., theta_z = 0.):

    # Construct rotation matrices
    if mesh.Dim == 3:
        Rx = np.array([[1., 0., 0.], [0., np.cos(theta_x), -np.sin(theta_x)], [0., np.sin(theta_x), np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y), 0., np.sin(theta_y)], [0., 1., 0.], [-np.sin(theta_y), 0., np.cos(theta_y)]])
        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0.], [np.sin(theta_z), np.cos(theta_z), 0.], [0., 0., 1.]])
        R = np.matmul(Rx, np.matmul(Ry, Rz))
    elif mesh.Dim == 2:
        R = np.array([[np.cos(theta_z), -np.sin(theta_z)], [np.sin(theta_z), np.cos(theta_z)]])
    else:
        raise NotImplementedError

    mesh.Coords = np.matmul(R, mesh.Coords.transpose()).transpose()

    print("Rotated nodes")


def VerifyPeriodicBoundary(mesh, BFG, icoord):
    coord = np.nan
    gbasis = mesh.gbasis
    for BF in BFG.BFaces:
        # Extract info
        ielem = BF.Elem
        face = BF.face

        ''' Get physical coordinates of face '''
        # Get local q = 1 nodes on face
        lfnodes = gbasis.get_local_face_principal_node_nums(mesh.gorder, face)

        # Convert to global node numbering
        # gfnodes = mesh.Elem2Nodes[ielem][lfnodes]

        # Physical coordinates of global nodes
        # coords = mesh.Coords[gfnodes]
        elem_coords = mesh.elements[ielem].node_coords
        coords = elem_coords[lfnodes]
        if np.isnan(coord):
            coord = coords[0, icoord]

        # Make sure coord1 matches for all nodes
        if np.any(np.abs(coords[:,icoord] - coord) > tol):
            raise ValueError

        # Now force each node to have the same exact coord1 
        coords[:,icoord] = coord

    return coord


def MatchBoundaryPair(mesh, which_dim, BFG1, BFG2, NodePairs, IdxInNodePairs, OldNode2NewNode, NewNodeOrder,
    NodePairsA = None, IdxInNodePairsA = None):
    '''
    NOTE: only q = 1 nodes are matched
    '''
    gbasis = mesh.gbasis

    if BFG1 is None and BFG2 is None:
        return

    NewNode2NewerNode = np.arange(mesh.nNode)
    NodesChanged = False

    if NodePairsA is not None:
        NodePairsB = NodePairsA.copy()
        if IdxInNodePairsA is None:
            raise ValueError
        else:
            IdxInNodePairsB = IdxInNodePairsA.copy()

    IFaces = mesh.IFaces
    icoord = which_dim

    # Extract relevant BFGs
    # BFG1 = None; BFG2 = None;
    # for BFG in mesh.BFaceGroups:
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
    # if icoord < 0 or icoord >= mesh.Dim:
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
    Remap NodePairs and IdxInNodePairs
    '''
    NewNodePairs = np.zeros_like(NodePairs, dtype=int) - 1
    IdxInNodePairs[:] = -1
    for i in range(len(NodePairs)):
        NewNodePairs[i,:] = OldNode2NewNode[NodePairs[i,:]]
    for i in range(len(NodePairs)):
        n1 = NewNodePairs[i,0]
        IdxInNodePairs[n1,0] = i; IdxInNodePairs[n1,1] = 0
        n2 = NewNodePairs[i,1]
        IdxInNodePairs[n2,0] = i; IdxInNodePairs[n2,1] = 1

    NodePairs = NewNodePairs

    # sanity check
    if np.amin(NewNodePairs) == -1:
        raise ValueError


    ''' 
    Identify and create periodic IFaces 
    '''
    for BFace1 in BFG1.BFaces:
        # Extract info
        ielem1 = BFace1.Elem
        face1 = BFace1.face

        ''' Get physical coordinates of face '''
        # Get local q = 1 nodes on face
        lfnodes1 = gbasis.get_local_face_principal_node_nums( 
            mesh.gorder, face1)
        nfnode = lfnodes1.shape[0]

        # Convert to global node numbering
        gfnodes1 = mesh.Elem2Nodes[ielem1][lfnodes1]
        nodesort1 = np.sort(gfnodes1)

        # Physical coordinates of global nodes
        coords1 = mesh.Coords[gfnodes1]

        # Pair each node with corresponding one on other boundary
        for BFace2 in BFG2.BFaces:
            # Extract info
            ielem2 = BFace2.Elem
            face2 = BFace2.face

            ''' Get physical coordinates of face '''
            # Get local q = 1 nodes on face
            lfnodes2 = gbasis.get_local_face_principal_node_nums(
                mesh.gorder, face2)

            # Convert to global node numbering
            gfnodes2 = mesh.Elem2Nodes[ielem2][lfnodes2]

            # Physical coordinates of global nodes
            coords2 = mesh.Coords[gfnodes2]

            ''' Check for complete match between all nodes '''
            nodesort2 = np.sort(gfnodes2)
            # Find nodes on boundary 2 paired with those in nodesort1
            idx1 = IdxInNodePairs[nodesort1, 0]
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
            #             for d in range(mesh.Dim):
            #                 if d == icoord: continue # skip periodic direction
            #                 # code.interact(local=locals())
            #                 coord2[d] = coord1[d]
            #                 mesh.Coords[gfnodes2[m],d] = coord1[d]
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
                    mesh.Coords[nodepairssort2] = mesh.Coords[nodepairs2]

                    # store for remapping elements to nodes later
                    NewNode2NewerNode[nodepairs2] = nodepairssort2

                    # Modify pairing as necessary
                    NodePairs[idx1,1] = nodepairssort2
                    IdxInNodePairs[nodepairssort2,0] = idx1

                    # Modify other pairings
                    if NodePairsA is not None:
                        for n in range(nfnode):
                            # check if node has been swapped
                            n2 = nodepairs2[n]
                            ns2 = nodepairssort2[n]
                            if n2 != ns2:
                                # check if it exists in other pairing
                                idxA = IdxInNodePairsA[n2,0]
                                if idxA != -1:
                                    b = IdxInNodePairsA[n2,1]
                                    NodePairsA[idxA, b] = ns2
                                    # reset n2
                                    IdxInNodePairsB[n2, :] = -1
                                    IdxInNodePairsB[ns2, 0] = idxA
                                    IdxInNodePairsB[ns2, 1] = b

                        # Put back into A
                        IdxInNodePairsA[:] = IdxInNodePairsB[:]


                    NodesChanged = True
                    # remap elements
                    for elem in range(mesh.nElem):
                        mesh.Elem2Nodes[elem, :] = NewNode2NewerNode[mesh.Elem2Nodes[elem, :]]
                    # reset NewNode2NewerNode
                    NewNode2NewerNode = np.arange(mesh.nNode)

                    
                # Create IFace between these two faces
                mesh.nIFace += 1
                IFaces.append(mesh_defs.IFace())
                IF = IFaces[-1]
                IF.ElemL = ielem1
                IF.faceL = face1
                IF.ElemR = ielem2
                IF.faceR = face2

                BFG1.nBFace -= 1
                BFG2.nBFace -= 1
                break




        if not match:
            raise Exception("Could not find matching boundary face")


    # Verification
    if BFG1.nBFace != 0 or BFG2.nBFace != 0:
        raise ValueError
    mesh.nBFaceGroup -= 2


    # if NodesChanged:
    #     # remap elements
    #     mesh = mesh.ElemGroups[0]
    #     nElem = mesh.nElem
    #     for elem in range(nElem):
    #         mesh.Elem2Nodes[elem, :] = NewNode2NewerNode[mesh.Elem2Nodes[elem, :]]

    # mesh.BFaceGroups.remove(BFG1)
    # mesh.BFaceGroups.remove(BFG2)
    mesh.BFaceGroups.pop(BFG1.name)
    mesh.BFaceGroups.pop(BFG2.name)

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

    NodePairs = np.zeros([mesh.nNode, 2], dtype=int) - 1
    IdxInNodePairs = np.zeros([mesh.nNode, 2], dtype=int) - 1
    nNodePair = 0
    node2_matched = [False]*mesh.nNode

    # Extract relevant BFGs
    # BFG1 = None; BFG2 = None;
    # for BFG in mesh.BFaceGroups:
    #     if BFG.Name == b1:
    #         BFG1 = BFG
    #     if BFG.Name == b2:
    #         BFG2 = BFG
    # BFG = None
    BFG1 = mesh.BFaceGroups[b1]
    BFG2 = mesh.BFaceGroups[b2]

    StartIdx = NextIdx

    # Sanity check
    # if BFG1 is None or BFG2 is None:
    #     raise Exception("One or both boundaries not found")
    # elif BFG1 == BFG2:
    #     raise Exception("Duplicate boundaries")

    icoord = which_dim
    if icoord < 0 or icoord >= mesh.Dim:
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
        elem = BFace.Elem
        face = BFace.face

        ''' Get physical coordinates of face '''
        # Get local q = 1 nodes on face
        lfnodes = gbasis.get_local_face_principal_node_nums( 
                mesh.gorder, face)

        # Convert to global node numbering
        gfnodes = mesh.Elem2Nodes[elem][lfnodes[:]]

        for node in gfnodes:
            if OldNode2NewNode[node] == -1: # has not been ordered yet
                OldNode2NewNode[node] = NextIdx
                NewNodeOrder[NextIdx] = node
                NextIdx += 1
            if IdxInNodePairs[node,0] == -1:
                NodePairs[nNodePair,0] = node
                IdxInNodePairs[node,0] = nNodePair
                IdxInNodePairs[node,1] = 0
                nNodePair += 1

    StopIdx = NextIdx

    '''
    Deal with second boundary
    '''

    for BFace in BFG2.BFaces:
        # Extract info
        elem = BFace.Elem
        face = BFace.face

        ''' Get physical coordinates of face '''
        # Get local q = 1 nodes on face
        lfnodes = gbasis.get_local_face_principal_node_nums( 
                mesh.gorder, face)

        # Convert to global node numbering
        gfnodes = mesh.Elem2Nodes[elem][lfnodes[:]]

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

            coord2 = mesh.Coords[node2]
            
            match = False
            for n in range(nNodePair):
                if NodePairs[n,1] != -1:
                    # node1 already paired - skip
                    continue
                node1 = NodePairs[n,0]

                coord1 = mesh.Coords[node1]
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
                        for d in range(mesh.Dim):
                            if d == icoord: continue # skip periodic direction
                            # code.interact(local=locals())
                            coord2[d] = coord1[d]
                    # Store node pair
                    idx1 = IdxInNodePairs[node1, 0]
                    NodePairs[idx1, 1] = node2
                    IdxInNodePairs[node2, 0] = idx1
                    IdxInNodePairs[node2, 1] = 1
                    # Flag node2 as matched
                    node2_matched[node2] = True
                    break

            # # Loop through the newly ordered periodic nodes on boundary 1
            # match = False
            # for idx in range(StopIdx-StartIdx):
            #     node1 = NewNodeOrder[StartIdx+idx]
            #     # If node1 already matched, skip
            #     if IdxInNodePairs[node1] != -1:
            #         continue
            #     coord1 = mesh.Coords[node1]
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
            #             for d in range(mesh.Dim):
            #                 if d == icoord: continue # skip periodic direction
            #                 # code.interact(local=locals())
            #                 coord2[d] = coord1[d]
            #         # Store node pair
            #         NodePairs[nNodePair] = np.array([node1, node2])
            #         IdxInNodePairs[node1] = nNodePair
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
    NodePairs = NodePairs[:nNodePair,:]

    # print
    if which_dim == 0:
        s = "x"
    elif which_dim == 1:
        s = "y"
    else:
        s = "z"
    print("Reordered periodic boundary in %s" % (s))

    return BFG1, BFG2, NodePairs, IdxInNodePairs, NextIdx


def RemapNodes(mesh, OldNode2NewNode, NewNodeOrder, NextIdx=-1):
    # nPeriodicNode = NextIdx
    # NewCoords = np.zeros_like(mesh.Coords)

    # Fill up OldNode2NewNode and NewNodeOrder with non-periodic nodes
    if NextIdx != -1:
        for node in range(mesh.nNode):
            if OldNode2NewNode[node] == -1: # has not been ordered yet
                OldNode2NewNode[node] = NextIdx
                NewNodeOrder[NextIdx] = node
                NextIdx += 1

    # New coordinate list
    # NewCoords = mesh.Coords[NewNodeOrder]
    mesh.Coords[:] = mesh.Coords[NewNodeOrder]

    # New Elem2Nodes
    NewElem2Nodes = np.zeros_like(mesh.Elem2Nodes, dtype=int) - 1
    nElem = mesh.nElem
    for elem in range(nElem):
        mesh.Elem2Nodes[elem,:] = OldNode2NewNode[mesh.Elem2Nodes[elem, :]]

    # Store in mesh
    # mesh.Coords = NewCoords
    # mesh.Elem2Nodes = NewElem2Nodes


def VerifyPeriodicMesh(mesh):
    # Loop through interior faces
    for IF in mesh.IFaces:
        elemL = IF.ElemL
        elemR = IF.ElemR
        faceL = IF.faceL
        faceR = IF.faceR
        gbasis = mesh.gbasis
        gorder = mesh.gorder

        ''' Get global face nodes - left '''
        fnodesL = gbasis.get_local_face_principal_node_nums( 
            gorder, faceL)

        # Convert to global node numbering
        fnodesL = mesh.Elem2Nodes[elemL][fnodesL[:]]

        ''' Get global face nodes - right '''
        fnodesR = gbasis.get_local_face_principal_node_nums( 
            gorder, faceR)

        # Convert to global node numbering
        fnodesR = mesh.Elem2Nodes[elemR][fnodesR[:]]

        ''' If exact same global nodes, then this is NOT a periodic face '''
        fnodesLsort = np.sort(fnodesL)
        fnodesRsort = np.sort(fnodesR)
        if np.all(fnodesLsort == fnodesRsort):
            # skip non-periodic face
            continue

        ''' Compare distances '''
        coordsL = mesh.Coords[fnodesLsort]
        coordsR = mesh.Coords[fnodesRsort]
        dists = np.linalg.norm(coordsL-coordsR, axis=1)
        if np.abs(np.max(dists) - np.min(dists)) > tol:
            raise ValueError        


def update_boundary_group_nums(mesh):
    i = 0
    for BFG in mesh.BFaceGroups.values():
        BFG.number = i
        i += 1

def MakePeriodicTranslational(mesh, x1=None, x2=None, y1=None, y2=None, z1=None, z2=None):

    ''' Reorder nodes '''
    OldNode2NewNode = np.zeros(mesh.nNode, dtype=int)-1  
        # OldNode2NewNode[n] = the new node number (post-reordering) of node n (pre-ordering)
    NewNodeOrder = np.zeros(mesh.nNode, dtype=int)-1
        # NewNodeOrder[i] = the node number (pre-reordering) of the ith node (post-reordering)
    NextIdx = 0

    # x
    BFGX1, BFGX2, NodePairsX, IdxInNodePairsX, NextIdx = ReorderPeriodicBoundaryNodes(mesh, 
        x1, x2, 0, OldNode2NewNode, NewNodeOrder, NextIdx)
    # y
    BFGY1, BFGY2, NodePairsY, IdxInNodePairsY, NextIdx = ReorderPeriodicBoundaryNodes(mesh, 
        y1, y2, 1, OldNode2NewNode, NewNodeOrder, NextIdx)
    # z
    BFGZ1, BFGZ2, NodePairsZ, IdxInNodePairsZ, NextIdx = ReorderPeriodicBoundaryNodes(mesh, 
        z1, z2, 2, OldNode2NewNode, NewNodeOrder, NextIdx)


    ''' Remap nodes '''
    RemapNodes(mesh, OldNode2NewNode, NewNodeOrder, NextIdx)


    ''' Match pairs of periodic boundary faces '''
    # x
    MatchBoundaryPair(mesh, 0, BFGX1, BFGX2, NodePairsX, IdxInNodePairsX, OldNode2NewNode, NewNodeOrder)
    # y
    MatchBoundaryPair(mesh, 1, BFGY1, BFGY2, NodePairsY, IdxInNodePairsY, OldNode2NewNode, NewNodeOrder, 
        NodePairsZ, IdxInNodePairsZ)
    # z
    MatchBoundaryPair(mesh, 2, BFGZ1, BFGZ2, NodePairsZ, IdxInNodePairsZ, OldNode2NewNode, NewNodeOrder)


    ''' Update face orientations '''
    # making it periodic messes things up
    # check_face_orientations(mesh)

    ''' Update boundary group numbers '''
    update_boundary_group_nums(mesh)

    ''' Verify valid mesh '''
    VerifyPeriodicMesh(mesh)

    ''' Update elements '''
    mesh.create_elements()









