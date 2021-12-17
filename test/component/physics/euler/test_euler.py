import numpy as np
import pytest
import sys
sys.path.append('../src')

import physics.euler.euler as euler

rtol = 1e-15
atol = 1e-15


def test_convective_flux_1D():
	'''
	This tests the convective flux for a 1D case.
	'''
	physics = euler.Euler1D()

	ns = physics.NUM_STATE_VARS

	P = 101325.
	rho = 1.1
	u = 2.5
	gamma = 1.4
	rhoE = P / (gamma - 1.) + 0.5 * rho * u * u

	Uq = np.zeros([1, 1, ns])

	irho, irhou, irhoE = physics.get_state_indices()

	Uq[:, :, irho] = rho
	Uq[:, :, irhou] = rho * u
	Uq[:, :, irhoE] = rhoE


	Fref = np.zeros([1, 1, ns, 1])
	Fref[:, :, irho, 0] = rho * u
	Fref[:, :, irhou, 0] = rho * u * u + P
	Fref[:, :, irhoE, 0] = (rhoE + P) * u

	physics.set_physical_params()
	F, (u2c, rhoc, pc) = physics.get_conv_flux_interior(Uq)

	np.testing.assert_allclose(rhoc, rho, rtol, atol)	
	np.testing.assert_allclose(u2c, u*u, rtol, atol)	
	np.testing.assert_allclose(pc, P, rtol, atol)	
	np.testing.assert_allclose(F, Fref, rtol, atol)


def test_convective_flux_1D_zero_velocity():
	'''
	This tests the convective flux for a 1D case but with zero vel
	'''
	physics = euler.Euler1D()

	ns = physics.NUM_STATE_VARS

	P = 101325.
	rho = 1.1
	gamma = 1.4
	rhoE = P / (gamma - 1.)

	Uq = np.zeros([1, 1, ns])

	irho, irhou, irhoE = physics.get_state_indices()

	Uq[:, :, irho] = rho
	Uq[:, :, irhou] = 0.
	Uq[:, :, irhoE] = rhoE


	Fref = np.zeros([1, 1, ns, 1])
	Fref[:, :, irho, 0] = 0.	
	Fref[:, :, irhou, 0] = P
	Fref[:, :, irhoE, 0] = 0.

	physics.set_physical_params()
	F, (u2c, rhoc, pc) = physics.get_conv_flux_interior(Uq)

	np.testing.assert_allclose(rhoc, rho, rtol, atol)	
	np.testing.assert_allclose(u2c, 0., rtol, atol)	
	np.testing.assert_allclose(pc, P, rtol, atol)	
	np.testing.assert_allclose(F, Fref, rtol, atol)


def test_convective_flux_2D():
	'''
	This tests the convective flux for a 2D case.
	'''
	physics = euler.Euler2D()

	ns = physics.NUM_STATE_VARS

	P = 101325.
	rho = 1.1
	u = 2.5
	v = 3.5
	gamma = 1.4
	rhoE = P / (gamma - 1.) + 0.5 * rho * (u * u + v * v)

	Uq = np.zeros([1, 1, ns])

	irho, irhou, irhov, irhoE = physics.get_state_indices()

	Uq[:, :, irho] = rho
	Uq[:, :, irhou] = rho * u
	Uq[:, :, irhov] = rho * v
	Uq[:, :, irhoE] = rhoE


	Fref = np.zeros([1, 1, ns, 2])
	Fref[:, :, irho, 0] = rho * u
	Fref[:, :, irhou, 0] = rho * u * u + P
	Fref[:, :, irhov, 0] = rho * u * v
	Fref[:, :, irhoE, 0] = (rhoE + P) * u
	Fref[:, :, irho, 1] = rho * v
	Fref[:, :, irhou, 1] = rho * u * v
	Fref[:, :, irhov, 1] = rho * v * v + P
	Fref[:, :, irhoE, 1] = (rhoE + P) * v

	physics.set_physical_params()
	F, (u2c, v2c, rhoc, pc) = physics.get_conv_flux_interior(Uq)

	np.testing.assert_allclose(rhoc, rho, rtol, atol)	
	np.testing.assert_allclose(u2c, u*u, rtol, atol)	
	np.testing.assert_allclose(v2c, v*v, rtol, atol)	
	np.testing.assert_allclose(pc, P, rtol, atol)	
	np.testing.assert_allclose(F, Fref, rtol, atol)


def test_convective_flux_2D_zero_velocity():
	'''
	This tests the convective flux for a 2D case with zero vel
	'''
	physics = euler.Euler2D()

	ns = physics.NUM_STATE_VARS

	P = 101325.
	rho = 1.1
	gamma = 1.4
	rhoE = P / (gamma - 1.)

	Uq = np.zeros([1, 1, ns])

	irho, irhou, irhov, irhoE = physics.get_state_indices()

	Uq[:, :, irho] = rho
	Uq[:, :, irhou] = 0.
	Uq[:, :, irhov] = 0.
	Uq[:, :, irhoE] = rhoE


	Fref = np.zeros([1, 1, ns, 2])
	Fref[:, :, irho, 0] = 0.
	Fref[:, :, irhou, 0] = P
	Fref[:, :, irhov, 0] = 0.
	Fref[:, :, irhoE, 0] = 0.
	Fref[:, :, irho, 1] = 0.
	Fref[:, :, irhou, 1] = 0.
	Fref[:, :, irhov, 1] = P
	Fref[:, :, irhoE, 1] = 0.

	physics.set_physical_params()
	F, (u2c, v2c, rhoc, pc) = physics.get_conv_flux_interior(Uq)

	np.testing.assert_allclose(rhoc, rho, rtol, atol)	
	np.testing.assert_allclose(u2c, 0., rtol, atol)	
	np.testing.assert_allclose(v2c, 0., rtol, atol)	
	np.testing.assert_allclose(pc, P, rtol, atol)	
	np.testing.assert_allclose(F, Fref, rtol, atol)


def test_conv_eigenvectors_multiplied_is_identity():
	'''
	This tests the convective eigenvectors in euler and ensures
	that when dotted together they are identity
	'''
	physics = euler.Euler1D()
	ns = physics.NUM_STATE_VARS
	irho, irhou, irhoE = physics.get_state_indices()
	physics.set_physical_params()
	U_bar = np.zeros([1, 1, ns])
	
	P = 101325.
	rho = 1.1
	u = 2.5
	gamma = 1.4
	rhoE = P / (gamma - 1.) + 0.5 * rho * u * u


	U_bar[:, :, irho] = rho
	U_bar[:, :, irhou] = rho*u
	U_bar[:, :, irhoE] = rhoE

	right_eigen, left_eigen = physics.get_conv_eigenvectors(U_bar)
	ldotr = np.einsum('elij,eljk->elik', left_eigen, right_eigen)

	expected = np.zeros_like(left_eigen)
	expected[:, :] = np.identity(left_eigen.shape[-1])

	np.testing.assert_allclose(ldotr, expected, rtol, atol)
