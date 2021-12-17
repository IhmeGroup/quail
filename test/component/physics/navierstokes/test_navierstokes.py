import numpy as np
import pytest
import sys
sys.path.append('../src')

import physics.navierstokes.navierstokes as navierstokes
import physics.navierstokes.tools as ns_tools

rtol = 1e-15
atol = 1e-15


def test_diffusion_flux_2D():
	'''
	This tests the diffusive flux for a 2D case.
	'''
	physics = navierstokes.NavierStokes2D()
	physics.set_physical_params()
	physics.get_transport = ns_tools.set_transport("Sutherland")

	ns = physics.NUM_STATE_VARS

	P = 101325.
	rho = 1.1
	u = 2.5
	v = 3.5
	gamma = physics.gamma
	R = physics.R
	rhoE = P / (gamma - 1.) + 0.5 * rho * (u * u + v * v)

	Uq = np.zeros([1, 1, ns])

	irho, irhou, irhov, irhoE = physics.get_state_indices()

	Uq[:, :, irho] = rho
	Uq[:, :, irhou] = rho * u
	Uq[:, :, irhov] = rho * v
	Uq[:, :, irhoE] = rhoE

	# Calculate viscosity and thermal conductivity
	mu, kappa = physics.get_transport(physics, Uq)
	nu = mu / rho

	# Get temperature
	T = physics.compute_variable("Temperature", Uq, 
		flag_non_physical=True)[0, 0, 0]

	np.random.seed(10)
	drdx = np.random.rand()
	drdy = np.random.rand()
	dudx = np.random.rand()
	dudy = np.random.rand()
	dvdx = np.random.rand() 
	dvdy = np.random.rand()
	divu = dudx + dvdy
	dTdx = np.random.rand()
	dTdy = np.random.rand()

	tauxx = mu * (dudx + dudx - 2. / 3. * divu)
	tauxy = mu * (dudy + dvdx)
	tauyy = mu * (dvdy + dvdy - 2. / 3. * divu)

	gUq = np.zeros([1, 1, ns, 2])

	gUq[:, :, irho, 0] = drdx
	gUq[:, :, irhou, 0] = drdx * u + rho * dudx
	gUq[:, :, irhov, 0] = drdx * v + rho * dvdx
	gUq[:, :, irhoE, 0] = (R * T / (gamma - 1.) + \
		0.5 * (u * u + v * v)) * drdx + rho * u * dudx \
		+ rho * v * dvdx + rho * R / (gamma - 1.) * dTdx

	gUq[:, :, irho, 1] = drdy
	gUq[:, :, irhou, 1] = drdy * u + rho * dudy
	gUq[:, :, irhov, 1] = drdy * v + rho * dvdy
	gUq[:, :, irhoE, 1] = (R * T / (gamma - 1.) + \
		0.5 * (u * u + v * v)) * drdy + rho * u * dudy \
		+ rho * v * dvdy + rho * R / (gamma - 1.) * dTdy

	Fref = np.zeros([1, 1, ns, 2])
	Fref[:, :, irhou, 0] = tauxx
	Fref[:, :, irhov, 0] = tauxy
	Fref[:, :, irhoE, 0] = tauxx * u + tauxy * v + kappa * dTdx

	Fref[:, :, irhou, 1] = tauxy
	Fref[:, :, irhov, 1] = tauyy
	Fref[:, :, irhoE, 1] = tauxy * u + tauyy * v + kappa * dTdy

	F = physics.get_diff_flux_interior(Uq, gUq)

	kappa = kappa[0, 0, 0]
	np.testing.assert_allclose(F, Fref, kappa*rtol, kappa*atol)


def test_diffusion_flux_2D_zero_velocity():
	'''
	This tests the diffusive flux for a 2D case with zero vel
	'''
	physics = navierstokes.NavierStokes2D()
	physics.set_physical_params()
	physics.get_transport = ns_tools.set_transport("Sutherland")

	ns = physics.NUM_STATE_VARS

	P = 101325.
	rho = 1.1

	gamma = physics.gamma
	R = physics.R
	rhoE = P / (gamma - 1.)

	Uq = np.zeros([1, 1, ns])

	irho, irhou, irhov, irhoE = physics.get_state_indices()

	Uq[:, :, irho] = rho
	Uq[:, :, irhou] = 0.
	Uq[:, :, irhov] = 0.
	Uq[:, :, irhoE] = rhoE

	# Calculate viscosity
	mu, kappa = physics.get_transport(physics, Uq)
	mu = mu[0,0,0]
	kappa = kappa[0,0,0]

	nu = mu / rho

	# Get temperature
	T = physics.compute_variable("Temperature", Uq, 
		flag_non_physical=True)[0, 0, 0]

	np.random.seed(10)
	drdx = np.random.rand()
	drdy = np.random.rand()

	dTdx = np.random.rand()
	dTdy = np.random.rand()

	gUq = np.zeros([1, 1, ns, 2])
	gUq[:, :, irho, 0] = drdx
	gUq[:, :, irhoE, 0] = (R * T / (gamma - 1.)) * drdx \
		+ rho * R / (gamma - 1.) * dTdx

	gUq[:, :, irho, 1] = drdy
	gUq[:, :, irhoE, 1] = (R * T / (gamma - 1.)) * drdy \
		+ rho * R / (gamma - 1.) * dTdy

	Fref = np.zeros([1, 1, ns, 2])
	Fref[:, :, irhoE, 0] = kappa * dTdx
	Fref[:, :, irhoE, 1] = kappa * dTdy

	F = physics.get_diff_flux_interior(Uq, gUq)
	np.testing.assert_allclose(F, Fref, kappa*rtol, kappa*atol)
