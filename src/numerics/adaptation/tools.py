import numpy as np

def calculate_A(phi, w):
    """Compute matrix A as defined in the documentation.
    """
    return np.einsum('js, jn, j -> jsn', phi, phi, w)

def calculate_B(phi, w):
    """Compute matrix B as defined in the documentation.
    """
    return np.einsum('js, j -> js', phi, w)
