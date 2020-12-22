import numpy as np

import meshing.meshbase as mesh_defs
import numerics.adaptation.tools as adapter_tools
import numerics.helpers.helpers as helpers

class Adapter():

    # -- Constant matrices -- #
    # Neighbor change matrix
    neighbor_change = np.array([
            [7, 0, 5],
            [6, 4, 1],
            [5, 2, 7],
            [4, 6, 3]])
    boundary_neighbor_change = np.array([
            [2, 0, 4],
            [2, 3, 1]])

    # Jacobian matrices for h-adaptation transformation
    T_J = np.empty((3, 2, 2, 2))
    T_J[0, 0] = np.array([[.5, 0.], [.5, 1.]])
    T_J[0, 1] = np.array([[1., .5], [0., .5]])
    T_J[1, 0] = np.array([[-1., -1.], [.5, 0.]])
    T_J[1, 1] = np.array([[-1., -1.], [1., .5]])
    T_J[2, 0] = np.array([[.5, 1.], [-1., -1.]])
    T_J[2, 1] = np.array([[0., .5], [-1., -1.]])

    T_const = np.empty((3, 2, 1, 2))
    T_const[0, 0] = np.array([[0., 0.]])
    T_const[0, 1] = np.array([[0., 0.]])
    T_const[1, 0] = np.array([[1., 0.]])
    T_const[1, 1] = np.array([[1., 0.]])
    T_const[2, 0] = np.array([[0., 1.]])
    T_const[2, 1] = np.array([[0., 1.]])

    T_J_inv = np.empty_like(T_J)
    for i in range(3):
        for j in range(2):
            T_J_inv[i, j] = np.linalg.inv(T_J[i, j])

    def __init__(self, solver, elem_to_adaptation_group = None,
            adaptation_groups = None):
        if elem_to_adaptation_group is None: elem_to_adaptation_group = {}
        if adaptation_groups is None: adaptation_groups = set()
        # Nodes in reference space
        self.xn_ref = solver.mesh.gbasis.get_nodes(solver.mesh.gbasis.order)
        # Quadrature points in reference space and quadrature weights
        # TODO: This may not work. You have to use order * 2 to do the quadrature
        # accurately for calculating the mass matrix, unsure if this has been
        # accounted for.
        self.xq_ref = solver.elem_helpers.quad_pts
        self.w = solver.elem_helpers.quad_wts.flatten()
        # Basis evaluated at quadrature points
        self.phi_xq_ref = solver.basis.get_values(self.xq_ref)
        # Reference gradient of geometric basis on reference quadrature points
        self.grad_phi_g = solver.mesh.gbasis.get_grads(self.xq_ref)
        # Adaptation groups
        self.elem_to_adaptation_group = elem_to_adaptation_group
        self.adaptation_groups = adaptation_groups
        # Precompute matrices
        self.A = adapter_tools.calculate_A(self.phi_xq_ref, self.w)
        self.B = adapter_tools.calculate_B(self.phi_xq_ref, self.w)

        # Figure out which quad points go in which element during coarsening
        self.xq_indices = [[[] for _ in range(3)] for _ in range(2)]
        for j in range(self.xq_ref.shape[0]):
            x, y = self.xq_ref[j]
            # For splitting face 0
            if y >= x:
                self.xq_indices[0][0].append(j)
            else:
                self.xq_indices[1][0].append(j)
            # For splitting face 1
            if y <= .5*(1 - x):
                self.xq_indices[0][1].append(j)
            else:
                self.xq_indices[1][1].append(j)
            # For splitting face 2
            if x >= .5*(1 - y):
                self.xq_indices[0][2].append(j)
            else:
                self.xq_indices[1][2].append(j)

    def adapt(self, solver):
        """Perform h-adaptation.
        """

        # Extract needed data from the solver object
        # TODO: Maybe wrap this?
        # The old function took as arguments:
        # dJ_old, Uc_old, iMM_old, neighbors_old, xn_old, coarsen_IDs,
        #         refine_IDs, split_face_IDs
        dJ_old = solver.elem_helpers.djac_elems[:, :, 0]
        Uc_old = solver.state_coeffs
        iMM_old = solver.elem_helpers.iMM_elems
        # TODO: Better way to do this. Either change how Quail stores neighbors
        # or change how this code uses them.
        neighbors_old = np.empty((Uc_old.shape[0], 3))
        for i in range(neighbors_old.shape[0]):
            neighbors_old[i] = solver.mesh.elements[i].face_to_neighbors
        xn_old = solver.mesh.node_coords[solver.mesh.elem_to_node_IDs]
        # TODO: Get an indicator. Right now, this is specific to the isentropic
        # vortex case.
        coarsen_IDs = set()
        #refine_IDs = np.array([0])
        #split_face_IDs = np.array([0])
        refine_IDs = set()
        Uq = helpers.evaluate_state(Uc_old, self.phi_xq_ref,
                skip_interp=solver.basis.skip_interp) # [ne, nq, ns]
        min_volume = 1
        for i in range(solver.mesh.num_elems):
            if np.any(Uq[i, :, 0] < .8) and solver.elem_helpers.vol_elems[i] > min_volume:
                refine_IDs.add(i)
                break
        refine_IDs = np.array(list(refine_IDs), dtype=int)
        split_face_IDs = np.empty(refine_IDs.size, dtype=int)
        for i, ID in enumerate(refine_IDs):
            lengths = np.empty(3)
            nodes = solver.mesh.node_coords[solver.mesh.elem_to_node_IDs[ID]]
            lengths[0] = np.linalg.norm(nodes[2] - nodes[1])
            lengths[1] = np.linalg.norm(nodes[0] - nodes[2])
            lengths[2] = np.linalg.norm(nodes[1] - nodes[0])
            split_face_IDs[i] = np.argmax(lengths)

        # == Coarsening == #

        # -- Loop through adaptation groups and perform coarsening -- #
        # Groups to be deleted
        delete_groups = []
        # Start counting number of elements from the old mesh
        n_elems_old = Uc_old.shape[0]
        n_elems_coarsened = n_elems_old
        # Set of elements that will be deleted by the coarsening
        delete_elems = set()
        # Loop over adaptation groups
        for group in self.adaptation_groups:
            # If not all members of the group need to be coarsened, then skip this
            # group
            if not all([elem_ID in coarsen_IDs for elem_ID in group.elem_IDs]):
                continue
            # Coarsen triangles
            self.coarsen_triangles(xn_old, iMM_old, dJ_old, Uc_old,
                    group, delete_groups)
            # Neighbors of the old triangles in the group
            old_neighbors = neighbors_old[group.elem_IDs]
            # Get exterior neighbor of each triangle (the neighbor that is
            # not one of the old triangles)
            exterior_neighbors = np.empty(group.elem_IDs.size)
            for i in range(exterior_neighbors.size):
                exterior_neighbors[i] = np.setdiff1d(old_neighbors[i], group.elem_IDs)[0]
            # Set neighbors of new elements
            for i in range(group.parent_elem_IDs.size):
                if group.parent_elem_IDs.size != 1:
                    other_side = group.parent_elem_IDs[1 - i]
                else:
                    other_side = -1
                neighbors_old[group.parent_elem_IDs[i],
                        group.face_pair[i]] = other_side
                neighbors_old[group.parent_elem_IDs[i],
                        group.face_pair[i] - 1] = exterior_neighbors[1 + 2*i]
                neighbors_old[group.parent_elem_IDs[i],
                        group.face_pair[i] - 2] = exterior_neighbors[2*i]
            # Decrease total count of elements
            n_elems_to_delete = int(len(group.elem_IDs) / 2)
            if n_elems_to_delete == 2:
                n_elems_coarsened -= 2
                delete_elems.update(group.elem_IDs[[1, 3]])
            else:
                n_elems_coarsened -= 1
                delete_elems.update(group.elem_IDs[[1]])

        # -- Array sizing and allocation -- #
        # Create new arrays
        xn_coarsened = np.empty((n_elems_coarsened,) + xn_old.shape[1:])
        neighbors_coarsened = np.empty((n_elems_coarsened,) + neighbors_old.shape[1:], dtype=int)
        dJ_coarsened = np.empty((n_elems_coarsened,) + dJ_old.shape[1:])
        iMM_coarsened = np.empty((n_elems_coarsened,) + iMM_old.shape[1:])
        Uc_coarsened = np.empty((n_elems_coarsened,) + Uc_old.shape[1:])
        # Loop over to reorder elements into these arrays
        i = 0
        for i_old in range(n_elems_old):
            # Only add elements that are not deleted
            if i_old not in delete_elems:
                xn_coarsened[i] = xn_old[i_old]
                neighbors_coarsened[i] = neighbors_old[i_old]
                dJ_coarsened[i] = dJ_old[i_old]
                iMM_coarsened[i] = iMM_old[i_old]
                Uc_coarsened[i] = Uc_old[i_old]
                i += 1

        # Delete groups that are no longer needed
        for group in delete_groups: self.adaptation_groups.remove(group)

        # == Refinement == #

        # -- Create refinement pairs of elements and faces -- #
        n_refined = refine_IDs.shape[0]
        # Create arrays
        elem_pairs = np.empty((n_refined, 2), dtype=int)
        face_pairs = np.empty(n_refined, dtype=(int, 2))
        if n_refined != 0:
            # The left element is always the one that was marked for refinement
            elem_pairs[:, 0] = refine_IDs
            face_pairs[:, 0] = split_face_IDs
            # The right element is the neighbor across the face being split
            elem_pairs[:, 1] = neighbors_old[refine_IDs, split_face_IDs]
            # The right face is whichever is neighboring the left element
            for i in range(n_refined):
                # Only do this for elements not refined at a boundary
                if elem_pairs[i, 1] != -1:
                    face_pairs[i, 1] = np.argwhere(neighbors_old[elem_pairs[i, 1]]
                            == elem_pairs[i, 0])
                # Otherwise, set to -1 to indicate boundary face
                else: face_pairs[i, -1] = -1

        # -- Array sizing and allocation -- #
        # Number of elements being refined at a boundary face. If the element
        # pair is -1, then it's a boundary refinement.
        n_refined_boundary = np.argwhere(elem_pairs == -1).shape[0]
        n_elems = n_elems_coarsened + 2*n_refined - n_refined_boundary
        # Create new arrays
        xn = np.empty((n_elems,) + xn_coarsened.shape[1:])
        neighbors = np.empty((n_elems,) + neighbors_coarsened.shape[1:], dtype=int)
        dJ = np.empty((n_elems,) + dJ_coarsened.shape[1:])
        iMM = np.empty((n_elems,) + iMM_coarsened.shape[1:])
        Uc = np.empty((n_elems,) + Uc_coarsened.shape[1:])
        # Copy old data into new arrays
        xn[:n_elems_coarsened] = xn_coarsened
        neighbors[:n_elems_coarsened, :] = neighbors_coarsened
        dJ[:n_elems_coarsened, :] = dJ_coarsened
        iMM[:n_elems_coarsened, :] = iMM_coarsened
        Uc[:n_elems_coarsened, :] = Uc_coarsened

        # -- Loop through refinement pairs and perform refinement -- #
        for i, (elem_pair, face_pair) in enumerate(zip(elem_pairs, face_pairs)):
            face_pair = tuple(face_pair)
            # Get IDs
            elem_L_ID, elem_R_ID = elem_pair
            face_L_ID, face_R_ID = face_pair
            new_elem_IDs = np.array([elem_L_ID, n_elems_old + i,
                    elem_R_ID, n_elems_old + i + 1])
            old_elem_IDs = np.array([elem_L_ID, elem_L_ID,
                    elem_R_ID, elem_R_ID])
            # For boundary refinement, only produce two elements
            if elem_pair[1] == -1:
                new_elem_IDs = new_elem_IDs[:2]
                old_elem_IDs = old_elem_IDs[:2]

            # Create new triangles
            self.refine_triangles(solver, xn, iMM, dJ, Uc, new_elem_IDs,
                    old_elem_IDs, face_pair)

            # Map possible neighbors
            if elem_R_ID != -1:
                possible_neighbors = np.array([
                        # Neighbor 0 is counterclockwise of the split face of elem L
                        neighbors[elem_L_ID, face_L_ID - 2],
                        # Neighbor 1 is clockwise of the split face of elem L
                        neighbors[elem_L_ID, face_L_ID - 1],
                        # Neighbor 2 is counterclockwise of the split face of elem R
                        neighbors[elem_R_ID, face_R_ID - 2],
                        # Neighbor 3 is clockwise of the split face of elem R
                        neighbors[elem_R_ID, face_R_ID - 1],
                        # Neighbor 4 is elem L
                        new_elem_IDs[0],
                        # Neighbor 5 is appended to the end
                        new_elem_IDs[1],
                        # Neighbor 6 is elem R
                        new_elem_IDs[2],
                        # Neighbor 7 is appended to the end
                        new_elem_IDs[3],
                        ])
                # Find new neighbors using this mapping
                neighbors[new_elem_IDs, :] = possible_neighbors[self.neighbor_change]
            else:
                possible_neighbors = np.array([
                        # Neighbor 0 is counterclockwise of the split face of elem L
                        neighbors[elem_L_ID, face_L_ID - 2],
                        # Neighbor 1 is clockwise of the split face of elem L
                        neighbors[elem_L_ID, face_L_ID - 1],
                        # Neighbor 2 is the boundary
                        -1,
                        # Neighbor 3 is elem L
                        new_elem_IDs[0],
                        # Neighbor 4 is appended to the end
                        new_elem_IDs[1],
                        ])
                # Find new neighbors using this mapping
                neighbors[new_elem_IDs, :] = possible_neighbors[self.boundary_neighbor_change]

            # Update neighbors of neighbors
            for j, elem_ID in enumerate(possible_neighbors[:2]):
                if elem_ID != -1:
                    # Find which face of the neighbor's neighbor used to be elem_L, then
                    # update it
                    neighbors[elem_ID, np.argwhere(neighbors[elem_ID] ==
                            elem_L_ID)] = new_elem_IDs[j]

        # TODO: Maybe wrap this?
        solver.elem_helpers.djac_elems = dJ[..., np.newaxis]
        solver.state_coeffs = Uc
        solver.elem_helpers.iMM_elems = iMM
        unique_nodes = []
        solver.mesh.node_coords = np.empty((solver.mesh.num_nodes,
            solver.mesh.ndims))
        solver.mesh.node_coords[:] = np.nan
        solver.mesh.elem_to_node_IDs = np.empty((solver.mesh.num_elems,
            solver.mesh.num_nodes_per_elem), dtype=int)
        next_ID = 0
        # Loop over elements
        for i in range(solver.mesh.num_elems):
            # Loop over nodes
            for j in range(solver.mesh.num_nodes_per_elem):
                # Check if this node exists yet
                node_ID = np.where(np.all(np.isclose(solver.mesh.node_coords,
                        xn[i, j]), axis=1))[0]
                node_doesnt_exist = node_ID.size == 0
                # If the node doesn't exist yet, then create it and add it to
                # the current element
                if node_doesnt_exist:
                    solver.mesh.node_coords[next_ID] = xn[i, j]
                    solver.mesh.elem_to_node_IDs[i, j] = next_ID
                    next_ID += 1
                # Otherwise, just use the ID that was found
                else:
                    solver.mesh.elem_to_node_IDs[i, j] = node_ID[0]
        # Create elements and update neighbors
        # TODO: Better way to do this
        solver.mesh.create_elements()
        for i in range(neighbors.shape[0]):
            solver.mesh.elements[i].face_to_neighbors = neighbors[i]
        # Reshape residual array
        solver.stepper.res = np.zeros_like(Uc)
        # TODO: Probably don't need to call all of this
        solver.elem_helpers.get_basis_and_geom_data(solver.mesh, solver.basis,
                solver.order)
        solver.elem_helpers.alloc_other_arrays(solver.physics, solver.basis,
                solver.order)

        return (xn, neighbors, n_elems, dJ, Uc, iMM)

    def refine_triangles(self, solver, xn, iMM, dJ, Uc, new_elem_IDs,
            old_elem_IDs, face_pair):
        # Get the element IDs in the group and update the number of elements and
        # nodes. This is different depending on whether a boundary is refined or
        # not.
        if old_elem_IDs.shape[0] == 4:
            group_old_elem_IDs = old_elem_IDs[[0, 2]]
            solver.mesh.num_elems += 2
            solver.mesh.num_nodes += 1 + 4 * (solver.mesh.gbasis.order - 1)
        else:
            group_old_elem_IDs = old_elem_IDs[[0]]
            solver.mesh.num_elems += 1
            solver.mesh.num_nodes += 1 + 3 * (solver.mesh.gbasis.order - 1)

        # Find the old face between the two elements of the pair by iterating
        # through and searching for it.
        # TODO: simpler way to do this
        middle_face = None
        for int_face in solver.mesh.interior_faces:
            if (int_face.elemL_ID in group_old_elem_IDs and int_face.elemR_ID in
                    group_old_elem_IDs):
                middle_face = int_face
                break

        # Create the four new faces
        # TODO: This won't work for boundary refinement
        solver.mesh.interior_faces.append(mesh_defs.InteriorFace(new_elem_IDs[0],
                2, new_elem_IDs[1], 1))
        solver.mesh.interior_faces.append(mesh_defs.InteriorFace(new_elem_IDs[1],
                0, new_elem_IDs[2], 0))
        solver.mesh.interior_faces.append(mesh_defs.InteriorFace(new_elem_IDs[2],
                1, new_elem_IDs[3], 2))
        solver.mesh.interior_faces.append(mesh_defs.InteriorFace(new_elem_IDs[3],
                0, new_elem_IDs[0], 0))

        # Create new adaptation group and add to set. If it's a boundary
        # refinement, then don't add the boundary (-1) elements/faces.
        new_group = AdaptationGroup(new_elem_IDs, group_old_elem_IDs,
                [self.elem_to_adaptation_group.get(ID) for ID in group_old_elem_IDs],
                iMM[group_old_elem_IDs], xn[group_old_elem_IDs],
                dJ[group_old_elem_IDs], face_pair, middle_face)
        self.adaptation_groups.add(new_group)
        self.elem_to_adaptation_group.update({ID : new_group for ID in new_elem_IDs})

        # Transform reference nodes into new nodes on reference element
        xn_ref_new = self.refinement_transformation(self.xn_ref, self.T_J, self.T_const, face_pair)
        xq_ref_new = self.refinement_transformation(self.xq_ref, self.T_J, self.T_const, face_pair)

        # The nodes of the new triangles, transformed from the parent reference
        # space to physical space. This requires eval'ing the geometric basis of the
        # parent element at the new points.
        # TODO: vectorize this
        gbasis_xn_ref_new = np.empty((4, solver.mesh.num_nodes_per_elem,
                solver.mesh.num_nodes_per_elem))
        for i in range(4):
            gbasis_xn_ref_new[i] = solver.mesh.gbasis.get_values(xn_ref_new[i])
        xn[new_elem_IDs] = np.einsum('ipn, inl -> ipl', gbasis_xn_ref_new,
                xn[old_elem_IDs])

        # Compute Jacobian matrix
        J = np.einsum('pnr, inl -> iplr', self.grad_phi_g, xn[new_elem_IDs])
        # Compute and store the Jacobian determinant
        dJ[new_elem_IDs] = np.linalg.det(J)

        # Compute and store inverse mass matrix
        iMM[new_elem_IDs] = np.linalg.inv(np.einsum('jsn, ij -> isn', self.A,
            dJ[new_elem_IDs]))

        # Old basis on new quad points
        # TODO: vectorize this
        phi_old_on_xq_new = np.empty((4, xq_ref_new.shape[1], solver.basis.nb))
        for i in range(4):
            phi_old_on_xq_new[i] = solver.basis.get_values(xq_ref_new[i])
        # Do L2 projection
        Uc[new_elem_IDs] = np.einsum('isn, js, ijt, itk, ij -> ink',
                iMM[new_elem_IDs], self.B, phi_old_on_xq_new, Uc[old_elem_IDs],
                dJ[new_elem_IDs])

    def coarsen_triangles(self, xn, iMM, dJ, Uc, group, delete_groups):
        # Get face pair of group being coarsened
        face_pair = group.face_pair
        # Get new element IDs from the previous parent
        new_elem_IDs = group.parent_elem_IDs
        # Get old element IDs from the elements in the group
        old_elem_IDs = group.elem_IDs
        # Get new xn, iMM, and dJ from the previously stored parent information
        xn[new_elem_IDs] = group.parent_xn
        iMM[new_elem_IDs] = group.parent_iMM
        dJ[new_elem_IDs] = group.parent_dJ
        # Remove old adaptation group from dict
        self.elem_to_adaptation_group.update({old_elem_ID : None for old_elem_ID
            in old_elem_IDs})
        # Mark old adaptation group for deletion
        delete_groups.append(group)
        # Update groups to be the same as the parent
        self.elem_to_adaptation_group.update({new_elem_IDs[i] :
            group.parent_groups[i] for i in range(new_elem_IDs.size)})

        U_new = np.empty((new_elem_IDs.size, self.xq_ref.shape[0]))
        # Loop over parent elements
        for i in range(new_elem_IDs.size):
            # Loop over the two elements within each parent
            for ii in range(2):
                # Indices of quadrature points which fit inside this subelement
                idx = self.xq_indices[ii][face_pair[i]]
                # Transform the quad points contained within this subelement
                xq_ref_new = np.einsum('lr, pr -> pl',
                        self.T_J_inv[(face_pair[i], ii)],
                        self.xq_ref[idx] - self.T_const[(face_pair[i], ii)])\

                # Evaluate basis on these points
                phi_xq_ref_new = numerics.calculate_phi(xq_ref_new)
                # Evaluate solution on these points
                U_new[i, idx] = np.einsum('jn, n', phi_xq_ref_new,
                        Uc[old_elem_IDs[2*i + ii]])

        # Do L2 projection
        Uc[new_elem_IDs] = np.einsum('isn, js, ij, ij -> in', iMM[new_elem_IDs],
                self.B, U_new, dJ[new_elem_IDs])

    def refinement_transformation(self, x_ref, T_J, T_const, face_pair):
        """Transform from the old element reference space to the two new elements
        during refinement.
        """
        if face_pair[1] != -1: n_elems_split = 2
        else: n_elems_split = 1
        x = np.empty((2 * n_elems_split, x_ref.shape[0], x_ref.shape[1]))
        for i in range(n_elems_split):
            x[2*i : 2*i + 2] = np.einsum('ilr, pr -> ipl', T_J[face_pair[i]],
                    x_ref) + T_const[face_pair[i]]
        return x


class AdaptationGroup():

    def __init__(self, elem_IDs, parent_elem_IDs, parent_groups, parent_iMM,
            parent_xn, parent_dJ, face_pair, middle_face):
        self.elem_IDs = elem_IDs
        self.parent_elem_IDs = parent_elem_IDs
        self.parent_groups = parent_groups
        self.parent_iMM = parent_iMM
        self.parent_xn = parent_xn
        self.parent_dJ = parent_dJ
        self.face_pair = face_pair
        self.middle_face = middle_face





# --------------------- OLD IMPLEMENTATION: TO BE REMOVED -------------------- #


#def adapt(solver, physics, mesh, stepper):
#    """Adapt the mesh by refining or coarsening it.
#    For now, this does very little - just splits an element in half.
#
#    Arguments:
#    solver - Solver object (solver/base.py)
#    physics - Physics object (physics/base.py)
#    mesh - Mesh object (meshing/meshbase.py)
#    stepper - Stepper object (timestepping/stepper.py)
#    """
#
#    # Calculate adaption criterion for each element
#    e = np.empty(mesh.num_elems)
#    for elem in mesh.elements:
#        # Get norm of gradient of each state variable at each quadrature point
#        gU = evaluate_gradient_norm(
#                solver.elem_operators.basis_phys_grad_elems[elem.id],
#                physics.U[elem.id,:,:])
#        # Get maximum density gradient
#        grho = np.max(gU[:,0])
#        # Multiply by mesh area
#        e[elem.id] = solver.elem_operators.vol_elems[elem.id] * grho
#    # Normalize the adaption criterion so it ranges from 0 to 1
#    e /= np.max(e)
#
#    # Print elements with high adaption criterion
#    for elem in mesh.elements:
#        if getattr(elem, 'deactivated', False): continue
#        if e[elem.id] > .4 : print(elem.id, e[elem.id])
#
#    # Array of flags for which elements to be split
#    needs_refinement = np.zeros(mesh.num_elems, dtype=bool)
#    # Split a random element within a range
#    closest_distance = 9e99
#    ref_range = 2
#    point = np.array([0,0]) + np.array([random.random()*ref_range - ref_range/2, random.random()*ref_range - ref_range/2])
#    for elem in mesh.elements:
#        distance = np.linalg.norm(np.mean(elem.node_coords, axis=0) - point)
#        if distance < closest_distance:
#            closest_distance = distance
#            closest_elem = elem
#    split_id = closest_elem.id
#    print(mesh.num_elems)
#    # For wedge case
#    #if mesh.num_elems == 3: split_id = 1
#    #elif mesh.num_elems == 7: split_id = 4
#    #elif mesh.num_elems == 11: split_id = 7
#    #else: split_id = 0
#    # For box test
#    #split_id = 0
#    # For isentropic vortex
#    #if mesh.num_elems == 50: split_id = [12,17,18,13]
#    #else: split_id = 12
#
#    split_id = np.argwhere(e > .3)
#    needs_refinement[split_id] = True
#
#    min_volume = .01
#
#    # Loop over all elements
#    for elem_id in range(mesh.num_elems):
#        # Only refine elements that need refinement
#        if needs_refinement[elem_id]:
#
#            # Get element
#            elem = mesh.elements[elem_id]
#
#            # If the element volume is below a threshold, skip it
#            if solver.elem_operators.vol_elems[elem.id] < min_volume: continue
#
#            # Skip deactivated elements
#            # TODO: add deactivated to constructor of Element?
#            if getattr(elem, 'deactivated', False): continue
#
#            # -- Figure out what face to split -- #
#            # Get info about the longest face of the element
#            long_face, long_face_node_ids = find_longest_face(elem, mesh)
#            # Get the midpoint of this face
#            midpoint = np.mean(mesh.node_coords[long_face_node_ids], axis=0)
#            # Add the midpoint as a new mesh node
#            mesh.node_coords = np.append(mesh.node_coords, [midpoint], axis=0)
#            midpoint_id = np.size(mesh.node_coords, axis=0) - 1
#            # Find which node on the long face is most counterclockwise (since
#            # nodes must be ordered counterclockwise)
#            ccwise_node_id, cwise_node_id = find_counterclockwise_node(elem.node_ids,
#                    long_face_node_ids[0], long_face_node_ids[1])
#            # Get neighbor across this face, if there is one
#            neighbor_id = elem.face_to_neighbors[long_face]
#            if neighbor_id != -1:
#                neighbor = mesh.elements[neighbor_id]
#                # Find the local ID of the long face on the neighbor
#                neighbor_opposing_node_id = np.setdiff1d(neighbor.node_ids, long_face_node_ids, assume_unique=True)[0]
#                neighbor_long_face = np.argwhere(neighbor.node_ids == neighbor_opposing_node_id)[0,0]
#
#            # Split this element
#            new_elem1, new_elem2 = split_element(mesh, physics, solver.basis, elem, long_face,
#                    midpoint_id, ccwise_node_id, cwise_node_id)
#            # Split the neighbor, making sure to flip clockwise/counterclockwise
#            if neighbor_id != -1:
#                new_elem3, new_elem4 = split_element(mesh, physics, solver.basis, neighbor, neighbor_long_face,
#                        midpoint_id, cwise_node_id, ccwise_node_id)
#
#            # Create the faces between the elements
#            append_face(mesh, new_elem2, new_elem1)
#            if neighbor_id != -1:
#                append_face(mesh, new_elem3, new_elem2)
#                append_face(mesh, new_elem4, new_elem3)
#                append_face(mesh, new_elem1, new_elem4)
#
#            # TODO: Figure out how to remove long face after making new ones
#            # "Deactivate" the original element and its neighbor
#            # TODO: figure out a better way to do this
#            elem.face_to_neighbors = np.array([-1,-1,-1])
#            if neighbor_id != -1:
#                neighbor.face_to_neighbors = np.array([-1,-1,-1])
#            offset = 1
#            corner = np.array([-10,-10])
#            elem.node_coords = np.array([corner, corner+[0,-offset], corner+[offset,0]])
#            if neighbor_id != -1:
#                neighbor.node_coords = np.array([corner+[0,-offset], corner+[offset,-offset], corner+[offset,0]])
#
#            # Call compute operators
#            solver.precompute_matrix_operators()
#            if solver.limiter is not None: solver.limiter.precompute_operators(solver)
#
#            # Delete residual
#            stepper.R = np.zeros_like(physics.U)
#
#            elem.deactivated = True
#            if neighbor_id != -1: neighbor.deactivated = True
#
#def split_element(mesh, physics, basis, elem, split_face, midpoint_id, ccwise_node_id, cwise_node_id):
#    """Split an element into two smaller elements.
#
#    Arguments:
#    mesh - Mesh object (meshing/meshbase.py)
#    physics - PhysicsBase object (physics/base/base.py)
#    basis - BasisBase object (numerics/basis/basis.py)
#    elem - Element object (meshing/meshbase.py) that needs to be split
#    split_face - local ID of face along which to split
#    midpoint_id - global ID of the node which is the midpoint of the split face
#    Returns:
#    (new_elem1, new_elem2) - tuple of newly created Element objects
#    """
#
#    # Node coords of ref element
#    ref_nodes = mesh.gbasis.get_nodes(mesh.gbasis.order)
#    # Nodes of new elements in reference element
#    ref_ccwise_node_id = (split_face - 1) % mesh.gbasis.nb
#    ref_cwise_node_id  = (split_face + 1) % mesh.gbasis.nb
#    ref_midpoint = np.mean(ref_nodes[[ref_ccwise_node_id, ref_cwise_node_id],:], axis=0)
#    # Nodes of new elements in parent reference
#    # TODO: Write a more general way to get high-order split elements in parent
#    # reference space - this assumes P2 elements!
#    ref_nodes_1 = p2_nodes(ref_midpoint, ref_nodes[ref_ccwise_node_id], ref_nodes[split_face])
#    ref_nodes_2 = p2_nodes(ref_midpoint, ref_nodes[split_face], ref_nodes[ref_cwise_node_id])
#    # Global node IDs of new elements
#    node_ids_1 = np.array([midpoint_id, ccwise_node_id, elem.node_ids[split_face]])
#    node_ids_2 = np.array([midpoint_id, elem.node_ids[split_face], cwise_node_id])
#
#    # Create first element
#    new_elem1 = append_element(mesh, physics, basis, node_ids_1, ref_nodes_1,
#            elem, split_face - 2, split_face, 2)
#    # Create second element
#    new_elem2 = append_element(mesh, physics, basis, node_ids_2, ref_nodes_2,
#            elem, split_face - 1, split_face, 1)
#    return new_elem1, new_elem2
#
#def append_element(mesh, physics, basis, node_ids, ref_nodes, parent, parent_face_id, split_face_id, new_split_face_id):
#    """Create a new element at specified nodes and append it to the mesh.
#    This function creates a new element and sets the neighbors of the element
#    and neighbor element across the face specified by face_id. The solution
#    array is interpolated from the parent element to the two new elements.
#
#    Arguments:
#    mesh - Mesh object (meshing/meshbase.py)
#    physics - PhysicsBase object (physics/base/base.py)
#    basis - BasisBase object (numerics/basis/basis.py)
#    node_ids - array of new element's node IDs
#    parent - Element object (meshing/meshbase.py), parent of the new element
#    parent_face_id - local ID of face in parent element which needs new neighbors
#    split_face_id - local ID of face in parent element which has been split
#    new_split_face_id - local ID of face in new element which has been split
#    Returns:
#    elem - Element object (meshing/meshbase.py), newly created element
#    """
#    # The index of node_ids which contains the midpoint of the split face is
#    # always 0
#    midpoint_index = 0
#    # Local ID of face in new element which needs new neighbors is always 0,
#    # since this face always opposes the midpoint, which is node 0
#    face_id = 0
#    # Wrap parent face ID to be positive
#    parent_face_id = parent_face_id % mesh.gbasis.NFACES
#    # Create element
#    mesh.elements.append(mesh_defs.Element())
#    elem = mesh.elements[-1]
#    # Set first element's id, node ids, coords, and neighbors
#    elem.id = len(mesh.elements) - 1
#    elem.node_ids = node_ids
#    elem.node_coords = mesh.node_coords[elem.node_ids]
#    elem.face_to_neighbors = np.full(mesh.gbasis.NFACES, -1)
#    # Append to element nodes in mesh
#    mesh.elem_to_node_ids = np.append(mesh.elem_to_node_ids, [node_ids], axis=0)
#    # Update number of elements
#    mesh.num_elems += 1
#    # Get parent's neighbor across face
#    parent_neighbor_id = parent.face_to_neighbors[parent_face_id]
#    # Add parent's neighbor to new elements's neighbor
#    elem.face_to_neighbors[face_id] = parent_neighbor_id
#    # If the parent's neighbor is not a boundary, update it
#    if parent_neighbor_id != -1:
#        # Get parent's neighbor
#        parent_neighbor = mesh.elements[parent_neighbor_id]
#        # Get index of face in parent's neighbor
#        parent_neighbor_face_index = np.argwhere(parent_neighbor.face_to_neighbors == parent.id)[0]
#        # Set new element as parent neighbor's neighbor
#        parent_neighbor.face_to_neighbors[parent_neighbor_face_index] = elem.id
#        # Update old face neighbors by looking for the face between parent and parent_neighbor
#        for face in mesh.interior_faces:
#            if face.elemL_id == parent.id and face.elemR_id == parent_neighbor.id:
#                face.elemL_id = elem.id
#                face.faceL_id = face_id
#                break
#            if face.elemR_id == parent.id and face.elemL_id == parent_neighbor.id:
#                face.elemR_id = elem.id
#                face.faceR_id = face_id
#                break
#    # If the parent's neighbor is a boundary, add a new boundary face
#    if parent_neighbor_id == -1:
#        # Search for the correct boundary group
#        found = False
#        for bgroup in mesh.boundary_groups.values():
#            for bface in bgroup.boundary_faces:
#                # If found, stop the search
#                if bface.elem_id == parent.id and bface.face_id == parent_face_id:
#                    found = True
#                    break
#            if found:
#                bgroup.boundary_faces.append(mesh_defs.BoundaryFace())
#                bgroup.boundary_faces[-1].elem_id = elem.id
#                bgroup.boundary_faces[-1].face_id = face_id
#                bgroup.num_boundary_faces += 1
#    # If the split face is a boundary, add a new boundary face
#    split_face_neighbor_id = parent.face_to_neighbors[split_face_id]
#    if split_face_neighbor_id == -1:
#        # Search for the correct boundary group
#        found = False
#        for bgroup in mesh.boundary_groups.values():
#            for bface in bgroup.boundary_faces:
#                # If found, stop the search
#                if bface.elem_id == parent.id and bface.face_id == split_face_id:
#                    found = True
#                    break
#            if found:
#                bgroup.boundary_faces.append(mesh_defs.BoundaryFace())
#                bgroup.boundary_faces[-1].elem_id = elem.id
#                bgroup.boundary_faces[-1].face_id = new_split_face_id
#                bgroup.num_boundary_faces += 1
#    # -- Set nodal solution values of new element -- #
#    # Evaluate basis functions at new nodes on parent reference element
#    basis_vals = basis.get_values(ref_nodes)
#    # Evaluate the state at these new nodes, and append to global solution
#    U_elem = numerics_helpers.evaluate_state(physics.U[parent.id,:,:], basis_vals)
#    physics.U = np.append(physics.U, [U_elem], axis=0)
#    return elem
#
#def append_face(mesh, elemL, elemR):
#    """Create a new face between two elements and append it to the mesh.
#
#    Arguments:
#    mesh - Mesh object (meshing/meshbase.py)
#    elemL - Element object, left of face (meshing/meshbase.py)
#    elemR - Element object, right of face (meshing/meshbase.py)
#    """
#    # Create the face between the elements
#    mesh.interior_faces.append(mesh_defs.InteriorFace())
#    mesh.interior_faces[-1].elemL_id = elemL.id
#    mesh.interior_faces[-1].elemR_id = elemR.id
#    # This assumes midpoint is node 0
#    mesh.interior_faces[-1].faceL_id = 2
#    mesh.interior_faces[-1].faceR_id = 1
#    # Set neighbors on either side of the face
#    elemL.face_to_neighbors[2] = elemR.id
#    elemR.face_to_neighbors[1] = elemL.id
#    # Update number of faces
#    mesh.num_interior_faces += 1
#
#def find_longest_face(elem, mesh):
#    """Find the longest face in an element.
#
#    Arguments:
#    elem - Element object (meshing/meshbase.py)
#    mesh - Mesh object (meshing/meshbase.py)
#    Returns:
#    (int, array[2]) - tuple of longest face ID and array of node IDs
#    """
#    # Arrays for area of each face and face nodes
#    face_areas = np.zeros(mesh.gbasis.NFACES)
#    face_node_ids = np.empty((mesh.gbasis.NFACES, 2), dtype=int)
#    # Loop over each face
#    for i in range(mesh.gbasis.NFACES):
#        # Get the neighbor across the face
#        face_neighbor = mesh.elements[elem.face_to_neighbors[i]]
#        # Calculate the face area and find face nodes
#        face_areas[i], face_node_ids[i,:] = face_geometry(elem, i,
#                mesh.node_coords)
#    # Get face with highest area
#    long_face = np.argmax(face_areas)
#    # Get node IDs of the longest face
#    long_face_node_ids = face_node_ids[long_face,:]
#    return (long_face, long_face_node_ids)
#
#def face_geometry(elem, face_id, node_coords):
#    """Find the area and nodes of a face.
#
#    Arguments:
#    elem - Element object (meshing/meshbase.py)
#    face_id - ID of face to get area/nodes of
#    node_coords - array of node coordinates, shape [num_nodes, dim]
#    Returns:
#    (float, array[2]) tuple of face area and node IDs
#    """
#    # Find the node IDs of the face. This works by removing the node
#    # opposite the face (only works for first-order triangles).
#    # TODO: replace this with gbasis.get_local_face_node_nums
#    face_node_ids = np.setdiff1d(elem.node_ids, elem.node_ids[face_id])
#    # Get the coordinates of these nodes
#    face_nodes = node_coords[face_node_ids,:]
#    # Return the area of the face (which is just the distance since this is a 2D
#    # code)
#    return (np.linalg.norm(face_nodes[0,:] - face_nodes[1,:]), face_node_ids)
#
#def find_counterclockwise_node(nodes, a, b):
#    """Find which of two neighboring nodes is more counterclockwise.
#    The function takes an array of nodes along with two neighboring nodes a and b, then
#    finds which of a and b are more counterclockwise. The nodes array is assumed
#    to be ordered counterclockwise, which means that whichever of a and b
#    appears later in the array (or appears at index 0) is the most counterclockwise.
#
#    Arguments:
#    nodes - array of node IDs
#    a - ID of the first node
#    b - ID of the second node
#    Returns
#    (int, int) tuple of counterclockwise node and clockwise node
#    """
#
#    # Find indices of a and b in the nodes array
#    a_index = np.argwhere(nodes == a)[0]
#    b_index = np.argwhere(nodes == b)[0]
#
#    # If a's index is higher
#    if a_index > b_index:
#        # If a is at the end and b is at the beginning, then b is ahead
#        if a_index == nodes.size-1 and b_index == 0:
#            ccwise_node = b
#            cwise_node = a
#        # In every other case, if a's index is higher then a is ahead
#        else:
#            ccwise_node = a
#            cwise_node = b
#    # If b's index is higher
#    else:
#        # If b is at the end and a is at the beginning, then a is ahead
#        if b_index == nodes.size-1 and a_index == 0:
#            ccwise_node = a
#            cwise_node = b
#        # In every other case, if b's index is higher then b is ahead
#        else:
#            ccwise_node = b
#            cwise_node = a
#    return (ccwise_node, cwise_node)
#
#def p2_nodes(a, b, c):
#    """Find nodes of a P2 element.
#
#    Arguments:
#    a - coordinates of first corner
#    b - coordinates of second corner
#    c - coordinates of third corner
#    Returns:
#    nodes - array, shape(6,2), containing P2 node coordinates
#    """
#    nodes = np.empty((6,2))
#    nodes[0,:] = a
#    nodes[1,:] = np.mean([a,b],axis=0)
#    nodes[2,:] = b
#    nodes[3,:] = np.mean([b,c],axis=0)
#    nodes[4,:] = c
#    nodes[5,:] = np.mean([c,a],axis=0)
#    return nodes
#
#def p1_nodes(a, b, c):
#    """Find nodes of a P1 element.
#
#    Arguments:
#    a - coordinates of first corner
#    b - coordinates of second corner
#    c - coordinates of third corner
#    Returns:
#    nodes - array, shape(3,2), containing P2 node coordinates
#    """
#    nodes = np.empty((3,2))
#    nodes[0,:] = a
#    nodes[1,:] = b
#    nodes[2,:] = c
#    return nodes
#
#def p0_nodes(a, b, c):
#    """Find nodes of a P0 element.
#
#    Arguments:
#    a - coordinates of first corner
#    b - coordinates of second corner
#    c - coordinates of third corner
#    Returns:
#    nodes - array, shape(3,2), containing P2 node coordinates
#    """
#    nodes = np.empty((1,2))
#    nodes[0,:] = (a+b+c)/3
#    return nodes
#
#def evaluate_gradient_norm(basis_val_grad, Uc):
#    """Evaluate the norm of the gradient at a set of points within an element.
#
#    Arguments:
#    basis_val_grad - array[nq,nb,dim] of basis gradients at each point for each
#            basis for each dimension
#    Uc - array[nb,ns] of element solution at each node
#    Returns:
#    array[nq,ns] of norm of gradient of each state variable at each point
#    """
#    gradU = np.empty((basis_val_grad.shape[0], Uc.shape[1], basis_val_grad.shape[2]))
#    # Loop over each dimension
#    for i in range(basis_val_grad.shape[2]):
#        gradU[:,:,i] = np.matmul(basis_val_grad[:,:,i], Uc)
#    # Return norm of gradient
#    return np.linalg.norm(gradU, axis=2)
