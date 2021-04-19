import numpy as np

import numerics.basis.tools as basis_tools
import meshing.meshbase as mesh_defs
import meshing.tools as mesh_tools
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
    T_J[1, 0] = np.array([[1., 0.], [0., .5]])
    T_J[1, 1] = np.array([[1., 0.], [-.5, .5]])
    T_J[2, 0] = np.array([[.5, -.5], [0., 1.]])
    T_J[2, 1] = np.array([[.5, 0.], [0., 1.]])

    T_const = np.empty((3, 2, 1, 2))
    T_const[0, 0] = np.array([[0., 0.]])
    T_const[0, 1] = np.array([[0., 0.]])
    T_const[1, 0] = np.array([[0., 0.]])
    T_const[1, 1] = np.array([[0., .5]])
    T_const[2, 0] = np.array([[.5, 0.]])
    T_const[2, 1] = np.array([[0., 0.]])

    T_J_inv = np.empty_like(T_J)
    for i in range(3):
        for j in range(2):
            T_J_inv[i, j] = np.linalg.inv(T_J[i, j])

    def __init__(self, solver, elem_to_adaptation_group = None,
            adaptation_groups = None):
        # TODO: Make work for 1D.
        if solver.mesh.ndims == 1: return
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
        self.face_quad_pts = solver.int_face_helpers.quad_pts
        # Basis evaluated at quadrature points
        self.phi_xq_ref = solver.basis.get_values(self.xq_ref)
        # Reference gradient of geometric basis on reference quadrature points
        self.grad_phi_g = solver.mesh.gbasis.get_grads(self.xq_ref)
        # Adaptation groups
        self.elem_to_adaptation_group = elem_to_adaptation_group
        self.adaptation_groups = adaptation_groups
        # Precompute matrices
        # TODO: Get rid of this
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
        num_nodes_old = solver.mesh.num_nodes
        dJ_old = solver.elem_helpers.djac_elems[:, :, 0]
        Uc_old = solver.state_coeffs
        iMM_old = solver.elem_helpers.iMM_elems

        # == Coarsening == #
        dJ_coarsened, iMM_coarsened, Uc_coarsened, n_elems_coarsened = \
                self.coarsen(solver, coarsen_IDs, iMM_old, dJ_old, Uc_old,
                refine_IDs)

        # == Refinement == #
        dJ, iMM, Uc, n_elems, num_new_interior_faces = \
                self.refine(solver.mesh.elements, solver.mesh.interior_faces,
                solver.mesh.gbasis, solver.basis, refine_IDs, split_face_IDs,
                dJ_coarsened, iMM_coarsened, Uc_coarsened, n_elems_coarsened)
        solver.mesh.num_elems = n_elems
        solver.mesh.num_interior_faces += num_new_interior_faces

        # == Interface to Quail == #
        solver.elem_helpers.djac_elems = dJ[..., np.newaxis]
        solver.state_coeffs = Uc
        solver.elem_helpers.iMM_elems = iMM
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
        solver.bface_helpers.store_neighbor_info(solver.mesh)
        solver.bface_helpers.get_basis_and_geom_data(solver.mesh, solver.basis,
                solver.order)

        return (n_elems, dJ, Uc, iMM)

    def refine(self, elements, interior_faces, gbasis, basis, refine_IDs,
            split_face_IDs, dJ_old, iMM_old, Uc_old, n_elems_old):
        # -- Array sizing and allocation -- #
        n_refined = refine_IDs.shape[0]
        n_elems = n_elems_old + n_refined
        # Create new arrays
        dJ = np.empty((n_elems,) + dJ_old.shape[1:])
        iMM = np.empty((n_elems,) + iMM_old.shape[1:])
        Uc = np.empty((n_elems,) + Uc_old.shape[1:])
        # Copy old data into new arrays
        dJ[:n_elems_old, :] = dJ_old
        iMM[:n_elems_old, :] = iMM_old
        Uc[:n_elems_old, :] = Uc_old

        # -- Loop through tagged elements and perform refinement -- #
        num_new_interior_faces = 0
        for i, (elem_ID, face_ID) in enumerate(zip(refine_IDs, split_face_IDs)):
            # Refinement always takes one element and turns it into two. The
            # element which retains the parent ID is called the left element.
            # The new element is called the right element.
            elemL_ID = elem_ID
            elemR_ID = n_elems_old + i
            # Create new elements and update faces
            num_new_interior_faces += self.refine_element(elements,
                    interior_faces, gbasis, basis, iMM, dJ, Uc, elemL_ID,
                    elemR_ID, face_ID)
        return dJ, iMM, Uc, n_elems, num_new_interior_faces

    def refine_element(self, elements, interior_faces, gbasis, basis, iMM, dJ,
            Uc, elem0_ID, elem1_ID, face_ID):
        """
        Split an element into two elements.
        """

        # Transform reference nodes into new nodes on reference element
        # TODO: Precompute these for each orientation
        xn_ref_new = self.refinement_transformation(self.xn_ref, face_ID)
        xq_ref_new = self.refinement_transformation(self.xq_ref, face_ID)

        # Get elem being split (which ends up being elem0)
        elem0 = elements[elem0_ID]
        # Store old faces and nodes
        old_faces = elem0.faces.copy()
        old_nodes = elem0.node_coords.copy()
        # Create new elem, elem1
        elem1 = mesh_defs.Element(elem1_ID)
        elements.append(elem1)

        # Get face being split
        face = elem0.faces[face_ID]

        # Orientation of adaptation
        forward = face.elemL_ID == elem0_ID

        # If needed, flip orientation of elements
        if not forward:
            xn_ref_new = xn_ref_new[::-1]
            xq_ref_new = xq_ref_new[::-1]

        num_nodes_per_elem = gbasis.get_num_basis_coeff(gbasis.order)

        # The geometric basis of the parent element evaluated at the new nodes
        gbasis_xn_ref_new = np.empty((2, num_nodes_per_elem,
                num_nodes_per_elem))
        for i in range(2):
            gbasis_xn_ref_new[i] = gbasis.get_values(xn_ref_new[i])
        # The nodes of the new triangles, transformed from the parent reference
        # space to physical space
        new_elem_IDs = [elem0_ID, elem1_ID]
        old_elem_IDs = [elem0_ID, elem0_ID]
        xn_new = np.einsum('ipn, nl -> ipl', gbasis_xn_ref_new, old_nodes)

        # Update data within the elements
        # TODO: Update the face tree?
        for i, elem in enumerate([elem0, elem1]):
            elem.node_coords = xn_new[i]

        # Figure out:
        # 1. the element ID of the neighbor across the face being split
        # 2. the local face ID on the neighbor of the face being split
        if forward:
            neighbor_ID = face.elemR_ID
            neighbor_face_ID = face.faceR_ID
        else:
            neighbor_ID = face.elemL_ID
            neighbor_face_ID = face.faceL_ID

        # If the face has no children, then new faces must be created
        create_new_faces = not face.children

        # If new faces need to be made
        if create_new_faces:
            # Make a face between elem0 and neighbor
            face0 = mesh_defs.InteriorFace()
            # Make a face between elem1 and neighbor
            face1 = mesh_defs.InteriorFace()
            # Add new faces to the mesh
            interior_faces.append(face0)
            interior_faces.append(face1)
            # Make new faces children of old face
            face.children = [face0, face1]

            # Get the reference Q1 nodes from endpoints of elem0's split face
            refQ1node_nums_split = gbasis.get_local_face_principal_node_nums(
                    gbasis.order, face_ID)
            refQ1nodes_split = gbasis.PRINCIPAL_NODE_COORDS[
                    refQ1node_nums_split]
            # If positive orientation
            if forward:
                # Q1 nodes on split side
                face0.refQ1nodes_L = refQ1nodes_split
                face1.refQ1nodes_L = refQ1nodes_split
                # Figure out the Q1 nodes on other side by cutting in half
                middle_point_R = np.mean(face.refQ1nodes_R, axis=0)
                face1.refQ1nodes_R = np.vstack((face.refQ1nodes_R[0],
                    middle_point_R))
                face0.refQ1nodes_R = np.vstack((middle_point_R,
                    face.refQ1nodes_R[1]))
            # If negative orientation
            else:
                # Q1 nodes on split side
                face0.refQ1nodes_R = refQ1nodes_split[::-1]
                face1.refQ1nodes_R = refQ1nodes_split[::-1]
                # Figure out the Q1 nodes on other side by cutting in half
                middle_point_L = np.mean(face.refQ1nodes_L, axis=0)
                face1.refQ1nodes_L = np.vstack((face.refQ1nodes_L[0],
                    middle_point_L))
                face0.refQ1nodes_L = np.vstack((middle_point_L,
                    face.refQ1nodes_L[1]))

            # Store old face IDs
            # TODO: Is this needed?
            face0.old_faceL_IDs.append(face0.faceL_ID)
            face0.old_faceR_IDs.append(face0.faceR_ID)
            face1.old_faceL_IDs.append(face1.faceL_ID)
            face1.old_faceR_IDs.append(face1.faceR_ID)

            # Update face neighbors
            # If positive orientation
            if forward:
                face0.elemL_ID = elem0_ID
                face0.elemR_ID = neighbor_ID
                face0.faceL_ID = face_ID
                face0.faceR_ID = neighbor_face_ID
                face1.elemL_ID = elem1_ID
                face1.elemR_ID = neighbor_ID
                face1.faceL_ID = face_ID
                face1.faceR_ID = neighbor_face_ID
            # If negative orientation
            else:
                face0.elemR_ID = elem0_ID
                face0.elemL_ID = neighbor_ID
                face0.faceR_ID = face_ID
                face0.faceL_ID = neighbor_face_ID
                face1.elemR_ID = elem1_ID
                face1.elemL_ID = neighbor_ID
                face1.faceR_ID = face_ID
                face1.faceL_ID = neighbor_face_ID
            # One face split into two, and the middle face will be made later,
            # so there are two new interior faces
            num_new_interior_faces = 2

        # If new faces do not need to be made
        else:
            # Get the faces from the face's children
            face0, face1 = face.children

            # Transform the Q1 nodes of each face on this side
            for pair_ID, (new_face, elem_ID) in enumerate(
                    zip([face0, face1], [elem0_ID, elem1_ID])):
                self.update_faces(new_face, face_ID, elem_ID, pair_ID, forward)

            # No faces were split into two, but the middle face will still be
            # made later, so there is one new interior face
            num_new_interior_faces = 1

        # If positive orientation
        if face.elemL_ID == elem0_ID:
            elemL = elem0
            elemR = elem1
            faceL = face0
            faceR = face1
            split_elemL_ID = 0
        # If negative orientation
        else:
            elemL = elem1
            elemR = elem0
            faceL = face1
            faceR = face0
            split_elemL_ID = 1
        # If negative orientation
        # Make a face between elem0 and elem1
        # NOTE: This assumes triangles
        middle_faceL_ID = (face_ID + 2) % 3
        middle_faceR_ID = (face_ID + 1) % 3
        middle_face = mesh_defs.InteriorFace(elemL.ID, middle_faceL_ID,
                elemR.ID, middle_faceR_ID)
        # Get the nodes from elemL's 2nd face
        face_node_nums = gbasis.get_local_face_node_nums(
                gbasis.order, middle_faceL_ID)
        xn_face_ref = xn_ref_new[split_elemL_ID, face_node_nums]
        geom_basis_face = gbasis.get_values(xn_face_ref)
        middle_face.node_coords = np.matmul(geom_basis_face, old_nodes)
        # Add reference Q1 nodes to face by getting them from the left and right
        # side
        refQ1node_nums_L = gbasis.get_local_face_principal_node_nums(
                gbasis.order, middle_faceL_ID)
        refQ1node_nums_R = gbasis.get_local_face_principal_node_nums(
                gbasis.order, middle_faceR_ID)
        middle_face.refQ1nodes_L = gbasis.PRINCIPAL_NODE_COORDS[
                refQ1node_nums_L]
        middle_face.refQ1nodes_R = gbasis.PRINCIPAL_NODE_COORDS[
                refQ1node_nums_R][::-1]

        # Add new face to the mesh
        interior_faces.append(middle_face)

        # Update element faces
        elemL.faces = np.roll([faceL, old_faces[face_ID - 2], middle_face],
                -face_ID)
        elemR.faces = np.roll([faceR, middle_face, old_faces[face_ID - 1]],
                -face_ID)

        # Update face neighbors
        # TODO: This is specific to triangles
        for local_face in old_faces:
            # Do not update the old face, which will be removed
            if local_face is not face:
                local_face_ID, L_or_R = adapter_tools.get_face_ID(local_face,
                        elem0_ID)
                # Update first face counterclockwise of split face
                if   ((face_ID + 1) % gbasis.NFACES) == local_face_ID:
                    adapter_tools.update_face_neighbor(local_face, elemL.ID,
                            face_ID + 1, L_or_R)
                # Update second face counterclockwise of split face
                elif ((face_ID + 2) % gbasis.NFACES) == local_face_ID:
                    adapter_tools.update_face_neighbor(local_face, elemR.ID,
                            face_ID + 2, L_or_R)

        # ---- Old stuff, before hanging nodes ---- #
        # Create new adaptation group and add to set. If it's a boundary
        # refinement, then don't add the boundary (-1) elements/faces.
        # TODO: middle face is removed later. Should it be copied before being
        # passed in, or no?
        #new_group = AdaptationGroup(new_elem_IDs, group_old_elem_IDs,
        #        [self.elem_to_adaptation_group.get(ID) for ID in group_old_elem_IDs],
        #        iMM[group_old_elem_IDs], xn[group_old_elem_IDs],
        #        dJ[group_old_elem_IDs], face_pair, middle_face, refined_faces)
        ## Add this group as child groups of its parent
        #for i in range(len(new_group.parent_groups)):
        #    if new_group.parent_groups[i] is not None:
        #        new_group.parent_groups[i].child_groups.append(new_group)
        #self.adaptation_groups.add(new_group)
        #self.elem_to_adaptation_group.update({ID : new_group for ID in new_elem_IDs})
        # ---- End old stuff, before hanging nodes ---- #

        # Remove the old face if new faces were made
        if create_new_faces: interior_faces.remove(face)

        # Compute Jacobian matrix
        J = np.einsum('pnr, inl -> iplr', self.grad_phi_g, xn_new)
        # Compute and store the Jacobian determinant
        dJ[new_elem_IDs] = np.linalg.det(J)

        # Compute and store inverse mass matrix
        iMM[new_elem_IDs] = np.linalg.inv(np.einsum('jsn, ij -> isn', self.A,
            dJ[new_elem_IDs]))

        # The basis functions of the parent element evaluated at the new quad
        # points
        phi_old_on_xq_new = np.empty((2, xq_ref_new.shape[1], basis.nb))
        for i in range(2):
            phi_old_on_xq_new[i] = basis.get_values(xq_ref_new[i])
        # Do L2 projection
        Uc[new_elem_IDs] = np.einsum('isn, js, ijt, itk, ij -> ink',
                iMM[new_elem_IDs], self.B, phi_old_on_xq_new, Uc[old_elem_IDs],
                dJ[new_elem_IDs])
        return num_new_interior_faces

    def coarsen_triangles(self, solver, iMM, dJ, Uc, group, delete_groups):
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

    def refinement_transformation(self, x_ref, face_ID):
        """Transform from the old element reference space to the two new elements
        during refinement.
        """
        x = np.einsum('ilr, pr -> ipl', self.T_J[face_ID],
                x_ref) + self.T_const[face_ID]
        return x

    def inverse_refinement_transformation(self, x_ref, face_ID, pair_ID):
        """Transform from the new element reference space to the old element.
        """
        x = np.einsum('lr, pr -> pl', self.T_J_inv[face_ID, pair_ID],
                x_ref - self.T_const[face_ID, pair_ID])
        return x

    def update_faces(self, face, face_ID, elem_ID, pair_ID, forward):
        """
        When refining a face that already has children, this function
        deep-updates the children recursively (children, children of children,
        etc.)
        """
        # Transform and update this face, depending on its orientation
        if forward:
            face.refQ1nodes_L = self.inverse_refinement_transformation(
                    face.refQ1nodes_L, face_ID, pair_ID)
            face.elemL_ID = elem_ID
            # TODO: Is it really necessary to update face_IDs? Shouldn't they
            # stay the same?
            face.faceL_ID = face_ID
        else:
            face.refQ1nodes_R = self.inverse_refinement_transformation(
                    face.refQ1nodes_R, face_ID, 1 - pair_ID)
            face.elemR_ID = elem_ID
            face.faceR_ID = face_ID
        # If it has children, transform and update them too
        if face.children:
            child0, child1 = face.children
            self.update_faces(child0, face_ID, elem_ID, pair_ID, forward)
            self.update_faces(child1, face_ID, elem_ID, pair_ID, forward)

    def coarsen(self, solver, coarsen_IDs, iMM_old, dJ_old, Uc_old, refine_IDs):
        """
        Loop through adaptation groups and perform coarsening.
        """
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
                self.coarsen_triangles(solver, iMM_old, dJ_old, Uc_old,
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
                dJ_coarsened[i] = dJ_old[i_old]
                iMM_coarsened[i] = iMM_old[i_old]
                Uc_coarsened[i] = Uc_old[i_old]
                i += 1
        # Update IDs in the faces, refine IDs, adaptation groups,
        # elem_to_adaptation_group after reordering
        # TODO: Add the same thing, but for boundary faces
        for i_old in range(n_elems_old):
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

        return dJ_coarsened, iMM_coarsened, Uc_coarsened, n_elems_coarsened


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
