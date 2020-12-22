import sys
sys.path.append('src')

import numpy as np
import pytest

rtol = 1e-15
atol = 1e-15


@pytest.mark.parametrize('order', [
    # -- Unit square -- #
    0, 1, 2, 3, 4, 5,
])
def test_lagrange_basis_should_be_nodal(lagrange_basis, order):
    # Evaluate Lagrange basis at nodes
    phi = lagrange_basis.get_values(lagrange_basis.get_nodes(order))
    # This matrix should be identity
    expected = np.identity(phi.shape[0])
    # Assert
    np.testing.assert_allclose(phi, expected, rtol, atol)
