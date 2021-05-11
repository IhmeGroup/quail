import numpy as np
import numerics.helpers.helpers as helpers


def custom_user_adaptation(solver):
    '''
    Split element 0 along face 0 after the first iteration.
    '''
    # Current iteration number
    i = int(solver.time / solver.stepper.dt) - 1

    if i < 2:
        refine_IDs = np.array([0])
        split_face_IDs = np.array([0])
        coarsen_IDs = set()
    else:
        refine_IDs = np.array([])
        split_face_IDs = np.array([])
        coarsen_IDs = set(range(solver.mesh.num_elems))

    return refine_IDs, split_face_IDs, coarsen_IDs
