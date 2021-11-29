import numpy as np
from external.optional_cantera import ct

GAS_CONSTANT = 8.3144621000e3 # [J / (K kmole)]

def set_state_from_conservatives(physics, elem_ID, quad_ID, Uq):

	if physics.gas is None:
		physics.gas = ct.Solution('air_test.yaml')

	irho, irhou, irhoE, irhoYO2 = physics.get_state_indices()

	# Get energy
	e = Uq[irhoE] / Uq[irho] - 0.5 * (Uq[irhou]**2 / Uq[irho])
	# Get specific volume
	nu = 1./Uq[irho]
	# Get YO2
	YO2 = Uq[irhoYO2] / Uq[irho]
	# Get YN2
	YN2 = 1.0 - Uq[irhoYO2] / Uq[irho]

	physics.gas.UVY = e, nu, "O2:{},N2:{}".format(YO2, YN2)


def set_state_from_primitives(physics, rho, P, u, Y):
	
	gas = physics.gas

	U = np.zeros([physics.NUM_STATE_VARS])
	irho, irhou, irhoE, irhoYO2 = physics.get_state_indices()

	U[irho] = rho
	U[irhou] = rho*u
	U[irhoYO2] = rho * Y[0]
	# U[irhoYN2] = rho * Y[1]

	W = get_W_from_Y(gas.molecular_weights, Y)
	T = get_T_from_rhop(rho, P, W)
	gas.TPY = T, P, "O2:{},N2:{}".format(Y[0, 0], Y[1, 0])

	gamma = get_gamma(gas.cv, W)
	
	# Double check this definition
	U[irhoE] = rho * gas.UV[0] + 0.5*rho*u*u

	return U

def get_W_from_Y(Wi, Y):
	Wi1 = 1./Wi
	return 1./np.dot(Wi1, Y)

def get_T_from_rhop(rho, P, W):
	return P / (rho * GAS_CONSTANT/W)

def get_gamma(cv, W):
	return (cv + GAS_CONSTANT / W) / cv;

def get_pressure(physics, Uq):
	gas_elems = physics.gas_elems
	ne = Uq.shape[0]
	nq = Uq.shape[1]

	P = np.zeros([ne, nq, 1])
	for ie in range(ne):
		for iq in range(nq):
			set_state_from_conservatives(physics, ie, iq, Uq[ie, iq])
			P[ie, iq] = physics.gas.P
	return P

def get_temperature(physics, Uq):
	gas_elems = physics.gas_elems
	ne = Uq.shape[0]
	nq = Uq.shape[1]

	T = np.zeros([ne, nq, 1])
	for ie in range(ne):
		for iq in range(nq):
			set_state_from_conservatives(physics, ie, iq, Uq[ie, iq])
			T[ie, iq] = physics.gas.T
	return T

def get_specificheatratio(physics, Uq):
	gas_elems = physics.gas_elems
	ne = Uq.shape[0]
	nq = Uq.shape[1]

	gamma = np.zeros([ne, nq, 1])
	for ie in range(ne):
		for iq in range(nq):
			set_state_from_conservatives(physics, ie, iq, Uq[ie, iq])
			gamma[ie, iq] = physics.gas.cp / \
				physics.gas.cv
	return gamma

def get_maxwavespeed(physics, Uq):

	gamma = get_specificheatratio(physics, Uq)
	P = get_pressure(physics, Uq)

	irho, irhou, irhoE, irhoYO2, irhoYN2 = physics.get_state_indices()
	smom = physics.get_momentum_slice()

	rho = Uq[:, :, [irho]]
	mom = Uq[:, :, smom]

	return np.linalg.norm(mom, axis=2, keepdims=True) / rho \
			+ np.sqrt(gamma * P / rho)
