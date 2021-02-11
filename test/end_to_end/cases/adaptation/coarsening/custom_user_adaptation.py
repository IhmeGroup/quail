import numpy as np
import numerics.helpers.helpers as helpers


def custom_user_adaptation(solver):
    '''
    This function refines the middle triangles on the first iteration, then
    coarsens them back afterwards.
    '''

    if solver.time < .02:
        refine_IDs = np.array([4])
        split_face_IDs = np.array([0])
        coarsen_IDs = set()
    else:
        refine_IDs = np.array([])
        split_face_IDs = np.array([])
        coarsen_IDs = {i for i in range(solver.mesh.num_elems)}

    return refine_IDs, split_face_IDs, coarsen_IDs
