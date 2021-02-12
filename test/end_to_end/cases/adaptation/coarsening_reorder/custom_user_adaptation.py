import numpy as np
import numerics.helpers.helpers as helpers


def custom_user_adaptation(solver):
    '''
    This function refines everywhere, then coarsens at the end, which induces an
    ID reordering.
    '''

    # Inputs - which elems and faces to split each iteration
    refine_IDs_total = np.array([5, 6])
    split_face_IDs_total = np.array([0, 0])

    # Current iteration number
    i = int(round(solver.time / solver.stepper.dt)) - 1

    # Split correct elem/face for current iteration
    if i < refine_IDs_total.size:
        refine_IDs = np.array([refine_IDs_total[i]])
        split_face_IDs = np.array([split_face_IDs_total[i]])
        coarsen_IDs = set()
    else:
    # If nothing left, then coarsen the right-most elements
        refine_IDs = np.array([])
        split_face_IDs = np.array([])
        coarsen_IDs = {5, 17, 24, 25}

    return refine_IDs, split_face_IDs, coarsen_IDs
