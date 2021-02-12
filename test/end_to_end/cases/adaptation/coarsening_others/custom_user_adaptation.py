import numpy as np
import numerics.helpers.helpers as helpers


def custom_user_adaptation(solver):
    '''
    This function refines the middle triangles several times, then
    coarsens them back afterwards.
    '''

    # Inputs
    max_distance = 1
    speed = 100000
    refine_point = [-.6 + speed*solver.time, -.3 + speed*solver.time]

    # Array of node coords of each element
    xn_old = solver.mesh.node_coords[solver.mesh.elem_to_node_IDs]
    elem_centers = np.mean(xn_old, axis=1)
    distances = elem_centers - np.array([refine_point])
    distance_norms = np.linalg.norm(distances, axis=1)
    # Find element closest to the point of refinement
    refine_ID = np.argmin(distance_norms)
    refine_IDs = np.array([refine_ID])

    # Coarsen elements that are far away from the point of refinement
    coarsen_IDs = set()
    for i, distance in enumerate(distance_norms):
        if distance > max_distance:
            coarsen_IDs.add(i)

    # Split along the longest face
    lengths = np.empty(3)
    nodes = xn_old[refine_ID]
    lengths[0] = np.linalg.norm(nodes[2] - nodes[1])
    lengths[1] = np.linalg.norm(nodes[0] - nodes[2])
    lengths[2] = np.linalg.norm(nodes[1] - nodes[0])
    split_face_IDs = np.array([np.argmax(lengths)])

    #if solver.time < .07:
    #    for i in range(solver.mesh.num_elems):
    #        if np.mean(np.linalg.norm(xn_old[i], axis=1)) < 2 and solver.elem_helpers.vol_elems[i] > min_volume:
    #            refine_IDs.add(i)
    #else:
    #    for i in range(solver.mesh.num_elems):
    #        coarsen_IDs.add(i)

    # TODO: This is a hack to only refine one at a time, since doing more is
    # untested.
    #if refine_IDs != set(): refine_IDs = {next(iter(refine_IDs))}

    return refine_IDs, split_face_IDs, coarsen_IDs
