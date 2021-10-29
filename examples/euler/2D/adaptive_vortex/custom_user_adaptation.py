import numpy as np
import meshing.meshbase as mesh_defs
import numerics.helpers.helpers as helpers


# This is a density-based adaptation sensor specific to the isentropic vortex
# case.
def custom_user_adaptation(solver):

    # Inputs
    min_volume = .1
    skip_iter = 5
    # Quick hack to stop adaptation
    #return np.array([]), np.array([]), set()

    # Current iteration number
    i_iter = int(round(solver.time / solver.stepper.dt)) - 1
    # Only adapt every skip_iter iterations
    if i_iter % skip_iter != 0:
        return np.array([]), np.array([]), set()

    coarsen_IDs = set()
    refine_IDs = set()
    Uc_old = solver.state_coeffs
    Uq = helpers.evaluate_state(Uc_old, solver.adapter.phi_xq_ref,
            skip_interp=solver.basis.skip_interp) # [ne, nq, ns]
    #if solver.time < .07:
    #    for i in range(solver.mesh.num_elems):
    #        if np.mean(np.linalg.norm(xn_old[i], axis=1)) < 2 and solver.elem_helpers.vol_elems[i] > min_volume:
    #            refine_IDs.add(i)
    #else:
    #    for i in range(solver.mesh.num_elems):
    #        coarsen_IDs.add(i)

    # Refine at the vortex, which is where density is low
    for i in range(solver.mesh.num_elems):
        # TODO: Hack to refine up to certain iteration
        #if i_iter > 4: break
        if np.any(Uq[i, :, 0] < .95) and solver.elem_helpers.vol_elems[i] > min_volume:
            refine_IDs.add(i)
            # Also add this element's neighbors
            for face in solver.mesh.elements[i].faces:
                # For interior faces
                try:
                    neighbors = [face.elemL_ID, face.elemR_ID]
                # For boundary faces
                except AttributeError:
                    neighbors = [face.elem_ID]
                # Check volume
                for neighbor_ID in neighbors:
                    if solver.elem_helpers.vol_elems[neighbor_ID] > min_volume:
                        refine_IDs.add(neighbor_ID)
    # Coarsen away from the vortex, which is where density is high
    for i in range(solver.mesh.num_elems):
        # TODO: Hack to coarsen all interior elements after certain iteration
        if i_iter < 5: break
        should_continue = False
        for face in solver.mesh.elements[i].faces:
            if isinstance(face, mesh_defs.BoundaryFace): should_continue = True
        if should_continue: continue
        coarsen_IDs.add(i)

        if np.all(Uq[i, :, 0] > .95):
            coarsen_IDs.add(i)

    # TODO: This is a hack to stop coarsening
    coarsen_IDs = set()

    # TODO: This is a hack to only refine one at a time
    #if refine_IDs != set(): refine_IDs = {next(iter(refine_IDs))}

    # TODO: This is a hack to refine the largest element that was marked, but
    # not elements on the boundary.
#    if refine_IDs != set():
#        largest_ID = next(iter(refine_IDs))
#        for ID in refine_IDs:
#            if (solver.elem_helpers.vol_elems[ID]
#                    > solver.elem_helpers.vol_elems[largest_ID]
#                    and not np.any(solver.mesh.elements[ID].node_coords < -2.9)
#                    and not np.any(solver.mesh.elements[ID].node_coords > 6.9)):
#                largest_ID = ID
#        refine_IDs = {largest_ID}

    # Convert refine_IDs to a Numpy array
    refine_IDs = np.array(list(refine_IDs), dtype=int)

    # Split along the longest face
    # TODO: Only works for Q1
    split_face_IDs = np.empty(refine_IDs.size, dtype=int)
    for i, ID in enumerate(refine_IDs):
        lengths = np.empty(3)
        nodes = solver.mesh.elements[ID].node_coords
        lengths[0] = np.linalg.norm(nodes[2] - nodes[1])
        lengths[1] = np.linalg.norm(nodes[0] - nodes[2])
        lengths[2] = np.linalg.norm(nodes[1] - nodes[0])
        split_face_IDs[i] = np.argmax(lengths)

    return refine_IDs, split_face_IDs, coarsen_IDs
