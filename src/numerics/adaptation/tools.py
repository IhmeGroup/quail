import numpy as np

import errors
import meshing.meshbase as mesh_defs

def calculate_A(phi, w):
    """
    Compute matrix A as defined in the documentation.
    """
    return np.einsum('js, jn, j -> jsn', phi, phi, w)

def calculate_B(phi, w):
    """
    Compute matrix B as defined in the documentation.
    """
    return np.einsum('js, j -> js', phi, w)

def get_face_ID(face, elem_ID):
    """
    Get local face ID of a face corresponding to the element ID.
    """
    # TODO: This function should be in meshbase, or something.
    # If it's an InteriorFace, check which side we're on
    if isinstance(face, mesh_defs.InteriorFace):
        if face.elemL_ID == elem_ID:
            return face.faceL_ID, True
        elif face.elemR_ID == elem_ID:
            return face.faceR_ID, False
        else:
            raise errors.DoesNotExistError
    # Otherwise, it's a BoundaryFace, and has only one side
    else:
        return face.face_ID, None

def update_face_neighbor(face, elem_ID, on_the_left):
    """
    Update neighbor of face given the element ID and the side of the face.
    """
    # TODO: This function should be in meshbase, or something.
    # If it's an InteriorFace, check which side we're on
    if isinstance(face, mesh_defs.InteriorFace):
        if on_the_left:
            face.elemL_ID = elem_ID
        else:
            face.elemR_ID = elem_ID
    # Otherwise, it's a BoundaryFace, and has only one side
    else:
        face.elem_ID = elem_ID

def get_new_node_IDs(start, old, n):
    """
    Get n new node IDs given the first one to start from as well as some old IDs
    to overwrite.
    """
    new_IDs = np.empty(n, dtype=int)
    new_IDs[:old.size] = old
    new_IDs[old.size:] = np.arange(start, start + n - old.size)
    return new_IDs
