import numpy as np
import pytest
import sys
sys.path.append('../src')

import numerics.basis.basis as basis_defs
import numerics.basis.tools as basis_tools

import solver.DG as DG
import meshing.meshbase as meshbase
import numerics.helpers.helpers as helpers


rtol = 1e-15
atol = 1e-15

def test_evaluate_gradient_returns_zero_when_state_is_constant():
	'''
	This test checks that the gradient of the state is zero when 
	the state is set to a constant for a Legendre basis.

	Note: We are working on a P1 reference element.
	'''
	basis_ref_grad = np.array([[0., 1.], [0., 1.]])
	basis_phys_grad_elems = basis_ref_grad.reshape([1, 2, 2, 1])

	Uc = np.zeros([1, 2, 4])

	# Setting first coefficient to 1 and the rest to zero sets 
	# the average of the state (for a modal basis)
	Uc[0, 0, :] = np.array([1., 2., 3., 4.])
	Uc[0, 1, :] = 0.
	gUq = helpers.evaluate_gradient(Uc, basis_phys_grad_elems)

	expected = np.zeros_like(gUq)
	np.testing.assert_allclose(gUq, expected, rtol, atol)

def test_evaluate_gradient_returns_one_when_state_is_linear():
	'''
	This test checks that the gradient of the state is one when 
	the state is set to a line for a Legendre basis.

	Note: We are working on a P1 reference element.
	'''
	basis_ref_grad = np.array([[0., 1.], [0., 1.]])
	basis_phys_grad_elems = basis_ref_grad.reshape([1, 2, 2, 1])

	Uc = np.zeros([1, 2, 4])

	Uc[0, 0, :] = 0.
	Uc[0, 1, :] = 1.
	gUq = helpers.evaluate_gradient(Uc, basis_phys_grad_elems)

	expected = np.ones_like(gUq)
	np.testing.assert_allclose(gUq, expected, rtol, atol)