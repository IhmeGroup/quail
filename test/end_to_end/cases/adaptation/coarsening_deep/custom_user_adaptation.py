import numpy as np
import numerics.helpers.helpers as helpers


def custom_user_adaptation(solver):
    '''
    This function refines the middle triangles several times, then
    coarsens them back afterwards.
    '''

    # Inputs - which elems and faces to split each iteration
    refine_IDs_total = np.array([4, 4, 4, 4, 4, 4,])
    split_face_IDs_total = np.array([0, 0, 0, 0, 0, 0,])

    # Current iteration number
    i = int(solver.time / solver.stepper.dt) - 1

    # Split correct elem/face for current iteration
    if i < refine_IDs_total.size:
        refine_IDs = np.array([refine_IDs_total[i]])
        split_face_IDs = np.array([split_face_IDs_total[i]])
        coarsen_IDs = set()
    else:
    # If nothing left, then coarsen everything
        refine_IDs = np.array([])
        split_face_IDs = np.array([])
        coarsen_IDs = {i for i in range(solver.mesh.num_elems)}

    return refine_IDs, split_face_IDs, coarsen_IDs
