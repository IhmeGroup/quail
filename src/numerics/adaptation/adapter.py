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

        # == Indicator == #

        # TODO: Implement some default indicators aside from custom ones
        # Run custom user adaptation function
        refine_IDs, split_face_IDs, coarsen_IDs =\
                solver.custom_user_adaptation(solver)
        # If no adaptation needs to happen, skip the rest of the function
        if (refine_IDs.size == 0 and split_face_IDs.size == 0 and
                len(coarsen_IDs) == 0):
            return

        # == Interface to Quail == #

        # Extract needed data from the solver object
        # TODO: Maybe wrap this?
        # The old function took as arguments:
        # dJ_old, Uc_old, iMM_old, neighbors_old, xn_old, coarsen_IDs,
        #         refine_IDs, split_face_IDs
        num_nodes_old = solver.mesh.num_nodes
        dJ_old = solver.elem_helpers.djac_elems[:, :, 0]
        Uc_old = solver.state_coeffs
        iMM_old = solver.elem_helpers.iMM_elems
        # TODO: Better way to do this. Either change how Quail stores neighbors
        # or change how this code uses them.
        neighbors_old = np.empty((Uc_old.shape[0], 3), dtype=int)
        for i in range(neighbors_old.shape[0]):
            neighbors_old[i] = solver.mesh.elements[i].face_to_neighbors
        xn_old = solver.mesh.node_coords[solver.mesh.elem_to_node_IDs]

        breakpoint()
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
            # Only do coarsening if all members of the group need to be
            # coarsened and the group has no child groups
            if (all([elem_ID in coarsen_IDs for elem_ID in group.elem_IDs])
                    and len(group.child_groups) == 0):
                print("Coarsening:")
                print(group.elem_IDs)

                # Coarsen triangles
                self.coarsen_triangles(solver, xn_old, iMM_old, dJ_old, Uc_old,
                        group, delete_groups)
                # Neighbors of the old triangles in the group
                old_neighbors = neighbors_old[group.elem_IDs]
                # Get exterior neighbor of each triangle (the neighbor that is
                # not one of the old triangles)
                exterior_neighbors = np.empty(group.elem_IDs.size, dtype=int)
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

                # Update old faces, by searching through and finding it
                # TODO: Add the reverse mapping (elem face ID to face object)
                new_elem_IDs = [
                        group.parent_elem_IDs[0],
                        group.parent_elem_IDs[0],
                        group.parent_elem_IDs[1],
                        group.parent_elem_IDs[1],
                        ]
                neighbor_face_IDs = [ID % 3 for ID in [
                        group.face_pair[0] + 1,
                        group.face_pair[0] + 2,
                        group.face_pair[1] + 1,
                        group.face_pair[1] + 2,
                        ]]
                for elem_ID, neighbor_ID, new_ID, neighbor_face_ID in zip(
                        group.elem_IDs, exterior_neighbors, new_elem_IDs,
                        neighbor_face_IDs):
                    for int_face in solver.mesh.interior_faces:
                        if (int_face.elemL_ID == elem_ID and int_face.elemR_ID ==
                                neighbor_ID):
                            int_face.elemL_ID = new_ID
                            int_face.faceL_ID = neighbor_face_ID
                        elif (int_face.elemL_ID == neighbor_ID and
                                int_face.elemR_ID == elem_ID):
                            int_face.elemR_ID = new_ID
                            int_face.faceR_ID = neighbor_face_ID

                # Update neighbors of neighbors
                #TODO: Wont work at a boundary
                for j, elem_ID in enumerate(exterior_neighbors):
                    if elem_ID != -1:
                        # Find which face of the neighbor's neighbor used to be elem_L, then
                        # update it
                        neighbors_old[elem_ID, np.argwhere(neighbors_old[elem_ID] ==
                                group.elem_IDs[j])] = new_elem_IDs[j]
                # Hack to only coarsen one group per iteration
                # TODO: get rid of this
                break

        # -- Array sizing and allocation -- #
        # Create new arrays
        xn_coarsened = np.empty((n_elems_coarsened,) + xn_old.shape[1:])
        neighbors_coarsened = np.empty((n_elems_coarsened,) + neighbors_old.shape[1:], dtype=int)
        dJ_coarsened = np.empty((n_elems_coarsened,) + dJ_old.shape[1:])
        iMM_coarsened = np.empty((n_elems_coarsened,) + iMM_old.shape[1:])
        Uc_coarsened = np.empty((n_elems_coarsened,) + Uc_old.shape[1:])

        # -- ID reordering -- #
        # Loop over to reorder elements into these arrays
        i = 0
        reordered_IDs = np.empty(n_elems_old, dtype=int)
        for i_old in range(n_elems_old):
            # Store the new ID after reordering
            reordered_IDs[i_old] = i
            # Only add elements that are not deleted
            if i_old not in delete_elems:
                xn_coarsened[i] = xn_old[i_old]
                neighbors_coarsened[i] = neighbors_old[i_old]
                dJ_coarsened[i] = dJ_old[i_old]
                iMM_coarsened[i] = iMM_old[i_old]
                Uc_coarsened[i] = Uc_old[i_old]
                i += 1
        # Update IDs in the neighbors, faces, refine IDs, adaptation groups,
        # elem_to_adaptation_group after reordering
        # TODO: Add the same thing, but for boundary faces
        for i_old in range(n_elems_old):
            neighbors_coarsened[neighbors_coarsened == i_old] = reordered_IDs[i_old]
            refine_IDs[refine_IDs == i_old] = reordered_IDs[i_old]
            for face in solver.mesh.interior_faces:
                if face.elemL_ID == i_old: face.elemL_ID = reordered_IDs[i_old]
                if face.elemR_ID == i_old: face.elemR_ID = reordered_IDs[i_old]
            # Update IDs in adaptation groups
            for group in self.adaptation_groups:
                group.elem_IDs[group.elem_IDs == i_old] = reordered_IDs[i_old]
                group.parent_elem_IDs[group.parent_elem_IDs == i_old] = reordered_IDs[i_old]
                for face in [group.middle_face, *group.refined_faces]:
                    if face.elemL_ID == i_old: face.elemL_ID = reordered_IDs[i_old]
                    if face.elemR_ID == i_old: face.elemR_ID = reordered_IDs[i_old]
            # Update elem_to_adaptation_group
            self.elem_to_adaptation_group[reordered_IDs[i_old]] = \
                    self.elem_to_adaptation_group.pop(i_old, None)

        # Delete groups that are no longer needed
        for group in delete_groups:
            # Remove this group as a child group of its parent
            for i in range(len(group.parent_groups)):
                if group.parent_groups[i] is not None:
                    group.parent_groups[i].child_groups.remove(group)
            # Remove the group
            self.adaptation_groups.remove(group)

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
            elem_pairs[:, 1] = neighbors_coarsened[refine_IDs, split_face_IDs]
            # The right face is whichever is neighboring the left element
            for i in range(n_refined):
                # Only do this for elements not refined at a boundary
                if elem_pairs[i, 1] != -1:
                    face_pairs[i, 1] = np.argwhere(neighbors_coarsened[elem_pairs[i, 1]]
                            == elem_pairs[i, 0])[0]
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
            new_elem_IDs = np.array([elem_L_ID, n_elems_coarsened + i,
                    elem_R_ID, n_elems_coarsened + i + 1])
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
            # Update elem_R if refinement wasn't at a boundary
            if new_elem_IDs.size == 4:
                for j, elem_ID in enumerate(possible_neighbors[2:4]):
                    if elem_ID != -1:
                        # Find which face of the neighbor's neighbor used to be elem_R, then
                        # update it
                        neighbors[elem_ID, np.argwhere(neighbors[elem_ID] ==
                                elem_R_ID)] = new_elem_IDs[j+2]
            # Update old faces, by searching through and finding it
            # TODO: Add the reverse mapping (elem face ID to face object)
            neighbor_face_IDs = [1, 2, 1, 2]
            for elem_ID, neighbor_ID, new_ID, neighbor_face_ID in zip(
                    old_elem_IDs, possible_neighbors, new_elem_IDs,
                    neighbor_face_IDs):
                for int_face in solver.mesh.interior_faces:
                    if (int_face.elemL_ID == elem_ID and int_face.elemR_ID ==
                            neighbor_ID):
                        int_face.elemL_ID = new_ID
                        int_face.faceL_ID = neighbor_face_ID
                    elif (int_face.elemL_ID == neighbor_ID and
                            int_face.elemR_ID == elem_ID):
                        int_face.elemR_ID = new_ID
                        int_face.faceR_ID = neighbor_face_ID

        # TODO: Maybe wrap this?
        solver.elem_helpers.djac_elems = dJ[..., np.newaxis]
        solver.state_coeffs = Uc
        solver.elem_helpers.iMM_elems = iMM
        node_coords_old = solver.mesh.node_coords.copy()
        solver.mesh.node_coords = 1e300 * np.ones((solver.mesh.num_nodes,
            solver.mesh.ndims), dtype=float)
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
            #if np.unique(solver.mesh.elem_to_node_IDs[i]).size != 3: breakpoint()
        #if np.unique(solver.mesh.node_coords, axis=0).shape[0] != 16:  breakpoint()
        #breakpoint()
        # Create elements and update neighbors
        # TODO: Better way to do this
        #breakpoint()
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
        solver.int_face_helpers.store_neighbor_info(solver.mesh)
        solver.int_face_helpers.get_basis_and_geom_data(solver.mesh,
                solver.basis, solver.order)

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
        refined_faces = [
                mesh_defs.InteriorFace(new_elem_IDs[0], 2, new_elem_IDs[1], 1),
                mesh_defs.InteriorFace(new_elem_IDs[1], 0, new_elem_IDs[2], 0),
                mesh_defs.InteriorFace(new_elem_IDs[2], 2, new_elem_IDs[3], 1),
                mesh_defs.InteriorFace(new_elem_IDs[3], 0, new_elem_IDs[0], 0),
                ]
        solver.mesh.interior_faces.append(refined_faces[0])
        solver.mesh.interior_faces.append(refined_faces[1])
        solver.mesh.interior_faces.append(refined_faces[2])
        solver.mesh.interior_faces.append(refined_faces[3])
        # Update number of faces - 4 new ones, 1 middle face removed
        solver.mesh.num_interior_faces += 3

        # Create new adaptation group and add to set. If it's a boundary
        # refinement, then don't add the boundary (-1) elements/faces.
        # TODO: middle face is removed later. Should it be copied before being
        # passed in, or no?
        new_group = AdaptationGroup(new_elem_IDs, group_old_elem_IDs,
                [self.elem_to_adaptation_group.get(ID) for ID in group_old_elem_IDs],
                iMM[group_old_elem_IDs], xn[group_old_elem_IDs],
                dJ[group_old_elem_IDs], face_pair, middle_face, refined_faces)
        # Add this group as child groups of its parent
        for i in range(len(new_group.parent_groups)):
            if new_group.parent_groups[i] is not None:
                new_group.parent_groups[i].child_groups.append(new_group)
        self.adaptation_groups.add(new_group)
        self.elem_to_adaptation_group.update({ID : new_group for ID in new_elem_IDs})

        # Remove middle face
        solver.mesh.interior_faces.remove(middle_face)

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

    def coarsen_triangles(self, solver, xn, iMM, dJ, Uc, group, delete_groups):
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
        # Update counts
        #TODO: Make this work at boundaries
        solver.mesh.num_elems -= 2
        solver.mesh.num_nodes -= 1 + 4 * (solver.mesh.gbasis.order - 1)
        solver.mesh.num_interior_faces -= 3
        # Removed old refined faces and add middle face back
        for face in group.refined_faces:
            solver.mesh.interior_faces.remove(face)
        solver.mesh.interior_faces.append(group.middle_face)

        U_new = np.empty((new_elem_IDs.size, self.xq_ref.shape[0],
                solver.physics.NUM_STATE_VARS))
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
                phi_xq_ref_new = solver.basis.get_values(xq_ref_new)
                # Evaluate solution on these points
                U_new[i, idx] = np.einsum('jn, nk -> jk', phi_xq_ref_new,
                        Uc[old_elem_IDs[2*i + ii]])

        # Do L2 projection
        Uc[new_elem_IDs] = np.einsum('isn, js, ijk, ij -> ink', iMM[new_elem_IDs],
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
            parent_xn, parent_dJ, face_pair, middle_face, refined_faces):
        self.elem_IDs = elem_IDs
        self.parent_elem_IDs = parent_elem_IDs
        self.parent_groups = parent_groups
        self.parent_iMM = parent_iMM
        self.parent_xn = parent_xn
        self.parent_dJ = parent_dJ
        self.face_pair = face_pair
        self.middle_face = middle_face
        self.refined_faces = refined_faces
        self.child_groups = []
