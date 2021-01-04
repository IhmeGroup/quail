import sys
sys.path.append('src')

import numpy as np
import pytest

import numerics.basis.basis as basis_defs

rtol = 1e-15
atol = 1e-15


@pytest.mark.parametrize('order', [
    # Order of Lagrange basis
    0, 1, 2, 3, 4, 5,
])
@pytest.mark.parametrize('Basis', [
    # Element geometry
    basis_defs.LagrangeSeg, basis_defs.LagrangeTri, basis_defs.LagrangeQuad,
])
def test_lagrange_basis_should_be_nodal(basis, order):
    '''
    The ith Lagrange basis should be equal to 1 at the ith node, and 0 at
    other nodes.
    '''
    # Evaluate Lagrange basis at nodes
    phi = basis.get_values(basis.get_nodes(order))
    # This matrix should be identity
    expected = np.identity(phi.shape[0])
    # Assert
    np.testing.assert_allclose(phi, expected, rtol, atol)