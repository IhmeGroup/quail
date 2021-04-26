import numpy as np
import pytest
import sys
sys.path.append('../src')

import numerics.basis.basis as basis_defs
import numerics.helpers.helpers as helpers

rtol = 1e-15
atol = 1e-15


@pytest.mark.parametrize('order', [
	# Order of Lagrange basis
	0, 1, 2, 3, 4, 5,
])
@pytest.mark.parametrize('Basis', [
	# Basis class representing the element geometry
	basis_defs.LagrangeSeg, basis_defs.LagrangeTri, basis_defs.LagrangeQuad,
])
@pytest.mark.parametrize('skip_interp', [
	# Order of Lagrange basis
	True, False,
])
def test_evaluate_state(basis, order, skip_interp):
	'''
	This test ensures that the ith Lagrange basis is equal to 1 at the ith
	node, and 0 at other nodes.

	Inputs:
	-------
		basis: Pytest fixture containing the basis object to be tested
		order: polynomial order of Lagrange basis being tested
	'''
	# Evaluate Lagrange basis at nodes
	phi = basis.get_values(basis.get_nodes(order))
	
	# Initialize Uc with ones
	ne = 10
	nb = phi.shape[0]
	ns = 4

	Uc = np.ones([ne, nb, ns])

	# This matrix should be identity
	expected = Uc

	# Evaluate state
	Uq = helpers.evaluate_state(Uc, phi, skip_interp)
	
	# Assert
	np.testing.assert_allclose(Uq, expected, rtol, atol)