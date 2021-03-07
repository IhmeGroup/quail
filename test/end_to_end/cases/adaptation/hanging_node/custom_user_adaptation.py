import numpy as np
import numerics.helpers.helpers as helpers


def custom_user_adaptation(solver):
    '''
    '''

    refine_IDs = np.array([1])
    split_face_IDs = np.array([0])
    coarsen_IDs = set()

    return refine_IDs, split_face_IDs, coarsen_IDs
