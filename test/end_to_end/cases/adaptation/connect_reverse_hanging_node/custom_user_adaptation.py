import numpy as np
import numerics.helpers.helpers as helpers


def custom_user_adaptation(solver):
    '''
    Split element 0, then split element 1 to connect back to the hanging node.
    '''
    # Current iteration number
    i = int(solver.time / solver.stepper.dt) - 1

    if i == 0:
        refine_IDs = np.array([0])
        split_face_IDs = np.array([0])
    elif i == 1:
        refine_IDs = np.array([1])
        split_face_IDs = np.array([0])
    else:
        refine_IDs = np.array([])
        split_face_IDs = np.array([])
    coarsen_IDs = set()

    return refine_IDs, split_face_IDs, coarsen_IDs
