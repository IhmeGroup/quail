import numpy as np
import pytest
import sys
sys.path.append('../src')

import physics.euler.euler as euler

rtol = 1e-15
atol = 1e-15


@pytest.mark.parametrize('conv_num_flux_type', [
	# Basis class
	"Roe", "LaxFriedrichs"
])
def test_numerical_flux_1D_consistency(conv_num_flux_type):
	'''
	This test ensures that the 1D numerical flux functions are 
	consistent, .e. F_numerical(u, u, n) = F(u) dot n
	'''
	physics = euler.Euler1D()
	physics.set_conv_num_flux(conv_num_flux_type)
	physics.set_physical_params()

	UqL = np.zeros([1, 1, physics.NUM_STATE_VARS])

	# Fill state
	P = 101325.
	rho = 1.1
	u = 2.5
	gamma = physics.gamma
	rhoE = P / (gamma - 1.) + 0.5 * rho * u * u

	irho, irhou, irhoE = physics.get_state_indices()

	UqL[:, :, irho] = rho
	UqL[:, :, irhou] = rho * u
	UqL[:, :, irhoE] = rhoE

	UqR = UqL.copy()

	# Normals
	normals = np.zeros([1, 1, 1])
	normals[:, :, 0] = 0.5

	# Compute numerical flux
	Fnum = physics.conv_flux_fcn.compute_flux(physics, UqL, UqR, normals)

	# Physical flux projected in normal direction
	F_expected, _ = physics.get_conv_flux_projected(UqL, normals)

	np.testing.assert_allclose(Fnum, F_expected, rtol, atol)


@pytest.mark.parametrize('conv_num_flux_type', [
	# Basis class
	"Roe", "LaxFriedrichs"
])
def test_numerical_flux_1D_conservation(conv_num_flux_type):
	'''
	This test ensures that the 1D numerical flux functions are 
	conservative, i.e. 
	F_numerical(uL, uR, n) = -F_numerical(uR, uL, -n)
	'''
	physics = euler.Euler1D()
	physics.set_conv_num_flux(conv_num_flux_type)
	physics.set_physical_params()

	UqL = np.zeros([1, 1, physics.NUM_STATE_VARS])
	UqR = UqL.copy()

	irho, irhou, irhoE = physics.get_state_indices()

	# Left state
	P = 101325.
	rho = 1.1
	u = 2.5
	gamma = physics.gamma
	rhoE = P / (gamma - 1.) + 0.5 * rho * u * u

	UqL[:, :, irho] = rho
	UqL[:, :, irhou] = rho * u
	UqL[:, :, irhoE] = rhoE

	# Right state
	P = 101325*2.
	rho = 0.7
	u = -3.5
	rhoE = P / (gamma - 1.) + 0.5 * rho * u * u

	UqR[:, :, irho] = rho
	UqR[:, :, irhou] = rho * u
	UqR[:, :, irhoE] = rhoE

	# Normals
	normals = np.zeros([1, 1, 1])
	normals[:, :, 0] = 0.5

	# Compute numerical flux
	Fnum = physics.conv_flux_fcn.compute_flux(physics, UqL, UqR, normals)

	# Compute numerical flux, but switch left and right states and negate 
	# normals
	F_expected = physics.conv_flux_fcn.compute_flux(physics, UqR, UqL, 
			-normals)

	np.testing.assert_allclose(Fnum, -F_expected, rtol, atol)


@pytest.mark.parametrize('conv_num_flux_type', [
	# Basis class
	"Roe", "LaxFriedrichs"
])
def test_numerical_flux_2D_consistency(conv_num_flux_type):
	'''
	This test ensures that the 2D numerical flux functions are 
	consistent, .e. F_numerical(u, u, n) = F(u) dot n
	'''
	physics = euler.Euler2D()
	physics.set_conv_num_flux(conv_num_flux_type)
	physics.set_physical_params()

	UqL = np.zeros([1, 1, physics.NUM_STATE_VARS])

	# Fill state
	P = 101325.
	rho = 1.1
	u = 2.5
	v = -3.5
	gamma = physics.gamma
	rhoE = P / (gamma - 1.) + 0.5 * rho * (u * u + v * v)

	irho, irhou, irhov, irhoE = physics.get_state_indices()

	UqL[:, :, irho] = rho
	UqL[:, :, irhou] = rho * u
	UqL[:, :, irhov] = rho * v
	UqL[:, :, irhoE] = rhoE

	UqR = UqL.copy()

	# Normals
	normals = np.zeros([1, 1, 2])
	normals[:, :, 0] = -0.4
	normals[:, :, 1] = 0.7

	# Compute numerical flux
	Fnum = physics.conv_flux_fcn.compute_flux(physics, UqL, UqR, normals)

	# Physical flux projected in normal direction
	F_expected, _ = physics.get_conv_flux_projected(UqL, normals)

	np.testing.assert_allclose(Fnum, F_expected, rtol, atol)


@pytest.mark.parametrize('conv_num_flux_type', [
	# Basis class
	"Roe", "LaxFriedrichs"
])
def test_numerical_flux_2D_conservation(conv_num_flux_type):
	'''
	This test ensures that the 2D numerical flux functions are 
	conservative, i.e. 
	F_numerical(uL, uR, n) = -F_numerical(uR, uL, -n)
	'''
	physics = euler.Euler2D()
	physics.set_conv_num_flux(conv_num_flux_type)
	physics.set_physical_params()

	UqL = np.zeros([1, 1, physics.NUM_STATE_VARS])
	UqR = UqL.copy()

	irho, irhou, irhov, irhoE = physics.get_state_indices()

	# Left state
	P = 101325.
	rho = 1.1
	u = 2.5
	v = -3.5
	gamma = physics.gamma
	rhoE = P / (gamma - 1.) + 0.5 * rho * (u * u + v * v)

	UqL[:, :, irho] = rho
	UqL[:, :, irhou] = rho * u
	UqL[:, :, irhov] = rho * v
	UqL[:, :, irhoE] = rhoE

	# Right state
	P = 101325*2.
	rho = 0.7
	u = -3.5
	v = -6.
	rhoE = P / (gamma - 1.) + 0.5 * rho * (u * u + v * v)

	UqR[:, :, irho] = rho
	UqR[:, :, irhou] = rho * u
	UqR[:, :, irhov] = rho * v
	UqR[:, :, irhoE] = rhoE

	# Normals
	normals = np.zeros([1, 1, 2])
	normals[:, :, 0] = -0.4
	normals[:, :, 1] = 0.7

	# Compute numerical flux
	Fnum = physics.conv_flux_fcn.compute_flux(physics, UqL, UqR, normals)

	# Compute numerical flux, but switch left and right states and negate 
	# normals
	F_expected = physics.conv_flux_fcn.compute_flux(physics, UqR, UqL, 
			-normals)

	np.testing.assert_allclose(Fnum, -F_expected, rtol, atol)
