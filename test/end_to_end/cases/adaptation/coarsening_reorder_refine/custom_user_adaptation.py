import numpy as np
import numerics.helpers.helpers as helpers


def custom_user_adaptation(solver):
    '''
    This function refines everywhere, then coarsens at the end, which induces an
    ID reordering.
    '''

    # Inputs - which elems and faces to split each iteration
    refine_IDs_total = [[5], [6], [], [25], []]
    split_face_IDs_total = [[0], [0], [], [2], []]
    coarsen_IDs_total = [set(), set(), {5, 17, 24, 25}, set(), {10, 25, 26, 27}]

    # Current iteration number
    i = int(round(solver.time / solver.stepper.dt)) - 1

    # Refine/coarsen correct elem/face for current iteration
    refine_IDs = np.array(refine_IDs_total[i])
    split_face_IDs = np.array(split_face_IDs_total[i])
    coarsen_IDs = coarsen_IDs_total[i]

    return refine_IDs, split_face_IDs, coarsen_IDs
