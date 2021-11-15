import numpy as np

GAS_CONSTANT = 8.3144621000e3 # [J / (K kmole)]

def set_state_from_conservatives(physics, elem_ID, quad_ID, Uq):

	irho, irhou, irhoE, irhoYO2, irhoYN2 = physics.get_state_indices()

	# Get energy
	e = Uq[elem_ID, quad_ID, irhoE] / Uq[elem_ID, quad_ID, irho]
	# Get specific volume
	nu = 1./Uq[elem_ID, quad_ID, irho]
	# Get YO2
	YO2 = Uq[elem_ID, quad_ID, irhoYO2] / Uq[elem_ID, quad_ID, irho]
	# Get YN2
	YN2 = Uq[elem_ID, quad_ID, irhoYN2] / Uq[elem_ID, quad_ID, irho]

	# gas_elem = physics.gas_elems[elem_ID, quad_ID]

	physics.gas_elems[elem_ID, quad_ID].UVY = e, nu, "O2:{},N2:{}".format(YO2, YN2)

def set_state_from_primitives(physics, rho, P, u, Y):
	
	gas = physics.gas

	U = np.zeros([physics.NUM_STATE_VARS])
	irho, irhou, irhoE, irhoYO2, irhoYN2 = physics.get_state_indices()

	U[irho] = rho
	U[irhou] = rho*u
	U[irhoYO2] = rho * Y[0]
	U[irhoYN2] = rho * Y[1]

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
	ne = gas_elems.shape[0]
	nq = gas_elems.shape[1]

	P = np.zeros_like(gas_elems)
	for ie in range(ne):
		for iq in range(nq):
			set_state_from_conservatives(physics, ie, iq, Uq)
			P[ie, iq] = physics.gas_elems[ie, iq].P
	return P

def get_temperature(physics, Uq):
	gas_elems = physics.gas_elems
	ne = gas_elems.shape[0]
	nq = gas_elems.shape[1]

	T = np.zeros_like(gas_elems)
	for ie in range(ne):
		for iq in range(nq):
			set_state_from_conservatives(physics, ie, iq, Uq)
			T[ie, iq] = physics.gas_elems[ie, iq].T
	return T

def get_specificheatratio(physics, Uq):
	gas_elems = physics.gas_elems
	ne = gas_elems.shape[0]
	nq = gas_elems.shape[1]

	gamma = np.zeros_like(gas_elems)
	for ie in range(ne):
		for iq in range(nq):
			set_state_from_conservatives(physics, ie, iq, Uq)
			gamma[ie, iq] = physics.gas_elems[ie, iq].cp / \
				physics.gas_elems[ie, iq].cv
	return gamma