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
        num_elems_old = solver.mesh.num_elems
        dJ_old = solver.elem_helpers.djac_elems[:, :, 0]
        Uc_old = solver.state_coeffs
        iMM_old = solver.elem_helpers.iMM_elems

        # == Coarsening == #
        dJ_coarsened, iMM_coarsened, Uc_coarsened = \
        self.coarsen(solver.mesh, solver.mesh.elements,
                solver.mesh.interior_faces, solver.mesh.boundary_groups,
                solver.mesh.gbasis, solver.basis, coarsen_IDs, refine_IDs,
                dJ_old, iMM_old, Uc_old, num_elems_old)

        # == Refinement == #
        # Refine
        dJ, iMM, Uc, n_elems = self.refine(solver.mesh.elements,
                solver.mesh.interior_faces, solver.mesh.boundary_groups,
                solver.mesh.gbasis, solver.basis, refine_IDs, split_face_IDs,
                dJ_coarsened, iMM_coarsened, Uc_coarsened, solver.mesh.num_elems)
        # Update counts
        solver.mesh.num_elems = n_elems
        solver.mesh.num_interior_faces = len(solver.mesh.interior_faces)
        for boundary_group_name in solver.mesh.boundary_groups:
            boundary_group = solver.mesh.boundary_groups[boundary_group_name]
            boundary_group.num_boundary_faces = len(boundary_group.boundary_faces)

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

    def refine(self, elements, interior_faces, boundary_groups, gbasis, basis,
            refine_IDs, split_face_IDs, dJ_old, iMM_old, Uc_old, n_elems_old):
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
        for i, (elem_ID, face_ID) in enumerate(zip(refine_IDs, split_face_IDs)):
            # Refinement always takes one element and turns it into two. The
            # element which retains the parent ID is called the left element.
            # The new element is called the right element.
            elemL_ID = elem_ID
            elemR_ID = n_elems_old + i
            # Create new elements and update faces
            # If splitting an interior face
            face = elements[elem_ID].faces[face_ID]
            if isinstance(face, mesh_defs.InteriorFace):
                self.refine_element(elements, interior_faces, interior_faces,
                        gbasis, basis, iMM, dJ, Uc, elemL_ID, elemR_ID, face_ID)
            # If splitting a boundary face
            else:
                self.refine_element(elements, interior_faces,
                        boundary_groups[face.name].boundary_faces, gbasis,
                        basis, iMM, dJ, Uc, elemL_ID, elemR_ID, face_ID)

        return dJ, iMM, Uc, n_elems

    def refine_element(self, elements, interior_faces, list_of_faces, gbasis, basis, iMM, dJ,
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
        # Set these elements as partners
        elem0.partner_ID = elem1_ID
        elem1.partner_ID = elem0_ID

        # Get face being split
        face = elem0.faces[face_ID]

        # Orientation of adaptation
        forward = adapter_tools.get_orientation(face, elem0_ID)

        # Whether or not we are refining a boundary face
        boundary = not isinstance(face, mesh_defs.InteriorFace)

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
        # This info only needed for interior faces.
        if not boundary:
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
            # Make a face between elem0 and neighbor, and another between
            # elem1 and neighbor
            if boundary:
                face0 = mesh_defs.BoundaryFace(face.name)
                face1 = mesh_defs.BoundaryFace(face.name)
            else:
                face0 = mesh_defs.InteriorFace()
                face1 = mesh_defs.InteriorFace()
            # Add new faces to the mesh
            list_of_faces.append(face0)
            list_of_faces.append(face1)
            # Make new faces children of old face
            face.children = [face0, face1]

            # Get the reference Q1 nodes from endpoints of elem0's split face
            refQ1node_nums_split = gbasis.get_local_face_principal_node_nums(
                    gbasis.order, face_ID)
            refQ1nodes_split = gbasis.PRINCIPAL_NODE_COORDS[
                    refQ1node_nums_split]
            # Boundary faces are always oriented forward and have only one side
            if boundary:
                face0.refQ1nodes = refQ1nodes_split
                face1.refQ1nodes = refQ1nodes_split
            # For interior faces, check orientation
            else:
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

            # Update boundary face neighbors
            if boundary:
                face0.elem_ID = elem0_ID
                face0.face_ID = face_ID
                face1.elem_ID = elem1_ID
                face1.face_ID = face_ID
            # Update interior face neighbors
            else:
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

        # If new faces do not need to be made
        else:
            # Get the faces from the face's children
            face0, face1 = face.children

            # Transform the Q1 nodes of each face on this side
            for pair_ID, (new_face, elem_ID) in enumerate(
                    zip([face0, face1], [elem0_ID, elem1_ID])):
                self.update_faces(new_face, face_ID, elem_ID, pair_ID, forward)

        # If positive orientation
        if forward:
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
                face_ID)
        elemR.faces = np.roll([faceR, middle_face, old_faces[face_ID - 1]],
                face_ID)

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
        if create_new_faces: list_of_faces.remove(face)

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

    def coarsen(self, mesh, elements, interior_faces, boundary_groups, gbasis, basis,
            coarsen_IDs, refine_IDs, dJ_old, iMM_old, Uc_old, n_elems_old):
        # List of elements that are removed
        removed_elem_IDs = []
        # -- Loop through tagged elements and perform coarsening -- #
        while coarsen_IDs:
            # Get element
            elem_ID = coarsen_IDs.pop()
            elem = elements[elem_ID]
            # Get partner element
            partner_ID = elem.partner_ID
            partner = elements[partner_ID]
            # Coarsening can only be performed if the partner element also needs
            # to be coarsened, and if both elements are leaf elements.
            if partner_ID in coarsen_IDs and elem.leaf and partner.leaf:
                # Remove partner from set
                coarsen_IDs.remove(partner_ID)

                # Refinement always takes one element and turns it into two. The
                # element which retains the parent ID is called the left element.
                # The new element is called the right element.
                #elemL_ID = elem_ID
                #elemR_ID = n_elems_old + i
                #TODO: This is a hack
                elemL_ID = elem_ID
                elemR_ID = partner_ID
                # Create new elements and update faces
                # If coarsening an interior face
                #face = elements[elem_ID].faces[face_ID]
                #if isinstance(face, mesh_defs.InteriorFace):
                removed_elem_IDs.append(
                        self.coarsen_element(mesh, elements, interior_faces, interior_faces,
                        gbasis, basis, iMM_old, dJ_old,
                        Uc_old, elemL_ID, elemR_ID))#, face_ID)
                # If coarsening a boundary face
                #else:
                #    self.coarsen_element(elements, interior_faces,
                #            boundary_groups[face.name].boundary_faces, gbasis,
                #            basis, iMM, dJ, Uc, elemL_ID, elemR_ID, face_ID)

        # Array sizing and allocation
        dJ_coarsened  = np.empty((mesh.num_elems,) + dJ_old.shape[1:])
        iMM_coarsened = np.empty((mesh.num_elems,) + iMM_old.shape[1:])
        Uc_coarsened  = np.empty((mesh.num_elems,) + Uc_old.shape[1:])

        # -- ID reordering -- #
        # Loop over to reorder elements into these arrays
        i = 0
        reordered_IDs = np.empty(n_elems_old, dtype=int)
        for i_old in range(n_elems_old):
            # Store the new ID after reordering
            reordered_IDs[i_old] = i
            # Only add elements that are not deleted
            if i_old not in removed_elem_IDs:
                dJ_coarsened[i] = dJ_old[i_old]
                iMM_coarsened[i] = iMM_old[i_old]
                Uc_coarsened[i] = Uc_old[i_old]
                i += 1
        # Update IDs in the faces, refine IDs after reordering
        for i_old in range(n_elems_old):
            refine_IDs[refine_IDs == i_old] = reordered_IDs[i_old]
            for face in interior_faces:
                if face.elemL_ID == i_old: face.elemL_ID = reordered_IDs[i_old]
                if face.elemR_ID == i_old: face.elemR_ID = reordered_IDs[i_old]
            for boundary_group in boundary_groups.values():
                for face in boundary_group.boundary_faces:
                    if face.elem_ID == i_old: face.elem_ID = reordered_IDs[i_old]

        return dJ_coarsened, iMM_coarsened, Uc_coarsened

    def coarsen_element(self, mesh, elements, interior_faces, list_of_faces, gbasis,
            basis, iMM_old, dJ_old, Uc_old, elem0_ID, elem1_ID):
        # Get new xn, iMM, and dJ from the previously stored parent information
        #xn[new_elem_IDs] = group.parent_xn
        #iMM[new_elem_IDs] = group.parent_iMM
        #dJ[new_elem_IDs] = group.parent_dJ
        elem0 = elements[elem0_ID]
        elem1 = elements[elem1_ID]
        # Get face ID of face between two elements being coarsened
        middle_face0_ID = np.argwhere(np.isin(elem0.faces, elem1.faces))[0, 0]
        middle_face1_ID = np.argwhere(elem1.faces == elem0.faces[middle_face0_ID])[0, 0]
        middle_face = elem0.faces[middle_face0_ID]
        # Figure out the refinement was forward or backward by inverting the
        # way faces were assigned during refinement
        # TODO: I am almost certain this is wrong - there is a 0 vs 1, an L vs
        # R, and the fact that i don't know which I pick to be 0 or 1 at the
        # start.
        if ((middle_face0_ID - 2) % 3) == ((middle_face1_ID - 1) % 3):
            forward = True
            face_ID = (middle_face0_ID - 2) % 3
        else:
            forward = False
            face_ID = (middle_face0_ID - 1) % 3
            # Swap element 0 and 1 since we are backwards
            elem0, elem1 = elem1, elem0
            elem0_ID, elem1_ID = elem1_ID, elem0_ID
        # Get nodes of new element
        xn_ref_new = self.inverse_refinement_transformation(self.xn_ref, face_ID, 0)
        # Convert to physical space
        xn = mesh_tools.ref_to_phys(mesh, elem0_ID, xn_ref_new)
        elem0.node_coords = xn
        # Update counts
        mesh.num_elems -= 1
        mesh.num_interior_faces -= 1

        # Compute Jacobian matrix
        J = np.einsum('pnr, nl -> plr', self.grad_phi_g, xn)
        # Compute the Jacobian determinant
        dJ = np.linalg.det(J)

        # Compute and store inverse mass matrix
        iMM = np.linalg.inv(np.einsum('jsn, j -> sn', self.A, dJ))

        U_new = np.empty((self.xq_ref.shape[0], Uc_old.shape[2]))
        # Loop over the two old elements
        for i, elem in enumerate([elem0, elem1]):
            # Indices of quadrature points which fit inside this subelement
            idx = self.xq_indices[i][face_ID]
            # Transform the quad points contained within this subelement
            xq_ref_new = self.inverse_refinement_transformation(
                    self.xq_ref[idx], face_ID, i)
            # Evaluate basis on these points
            phi_xq_ref_new = basis.get_values(xq_ref_new)
            # Evaluate solution on these points
            U_new[idx] = np.einsum('jn, nk -> jk', phi_xq_ref_new,
                    Uc_old[elem.ID])

        # Do L2 projection
        Uc_old[elem0_ID] = np.einsum('sn, js, jk, j -> nk', iMM, self.B, U_new, dJ)

        # Update face neighbors
        # TODO: This is specific to triangles
        for local_face in elem1.faces:
            # Do not update the middle face, which will be removed
            if local_face is not middle_face:
                local_face_ID, L_or_R = adapter_tools.get_face_ID(local_face,
                        elem1_ID)
                adapter_tools.update_face_neighbor(local_face, elem0_ID,
                        local_face_ID, L_or_R)
        # Remove middle face
        list_of_faces.remove(middle_face)

        # Return the element ID that should be removed
        return elem1_ID

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
