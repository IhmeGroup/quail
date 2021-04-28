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

def update_face_neighbor(face, elem_ID, face_ID, on_the_left):
    """
    Recursively update neighbors of faces given the element ID and the side of
    the face.
    """
    # TODO: This function should be in meshbase, or something.
    # If it's an InteriorFace, check which side we're on
    if isinstance(face, mesh_defs.InteriorFace):
        if on_the_left:
            face.elemL_ID = elem_ID
            face.faceL_ID = face_ID
        else:
            face.elemR_ID = elem_ID
            face.faceR_ID = face_ID
    # Otherwise, it's a BoundaryFace, and has only one side
    else:
        face.elem_ID = elem_ID
        face.face_ID = face_ID
    # Recursively update child faces
    if face.children:
        child0, child1 = face.children
        update_face_neighbor(child0, elem_ID, face_ID, on_the_left)
        update_face_neighbor(child1, elem_ID, face_ID, on_the_left)

def get_orientation(face, elem_ID):
    """
    Get orientation of a face with respect to a given element ID.
    """
    if isinstance(face, mesh_defs.InteriorFace):
        return face.elemL_ID == elem_ID
    else:
        return True
