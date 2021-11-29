import numpy as np
from external.optional_cantera import ct
import ctypes

import general

GAS_CONSTANT = 8.3144621000e3 # [J / (K kmole)]

LIB = ctypes.cdll.LoadLibrary(general.cantera_lib)


def set_state_from_conservatives(physics, elem_ID, quad_ID, Uq):

	if physics.gas is None:
		physics.gas = ct.Solution(physics.CANTERA_FILENAME)

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
	ne = Uq.shape[0]
	nq = Uq.shape[1]
	ns = Uq.shape[-1]
	nsp = physics.NUM_SPECIES

	filename = physics.c_cantera_file()

	P = np.zeros([ne, nq, 1])
	LIB.get_pressure(
		ctypes.c_void_p(Uq.ctypes.data), 
		ctypes.c_void_p(P.ctypes.data),
		ctypes.c_int(ne), 
		ctypes.c_int(nq), 
		ctypes.c_int(ns),
		ctypes.c_int(nsp),
		ctypes.c_int(physics.NDIMS),
		physics.c_cantera_file()
		)
	return P


def get_temperature(physics, Uq):
	ne = Uq.shape[0]
	nq = Uq.shape[1]
	ns = Uq.shape[-1]
	nsp = physics.NUM_SPECIES

	T = np.zeros([ne, nq, 1])
	LIB.get_temperature(
		ctypes.c_void_p(Uq.ctypes.data), 
		ctypes.c_void_p(T.ctypes.data),
		ctypes.c_int(ne), 
		ctypes.c_int(nq), 
		ctypes.c_int(ns),
		ctypes.c_int(nsp),
		ctypes.c_int(physics.NDIMS),
		physics.c_cantera_file()
			)
	return T


def get_specificheatratio(physics, Uq):
	ne = Uq.shape[0]
	nq = Uq.shape[1]
	ns = Uq.shape[-1]
	nsp = physics.NUM_SPECIES
	
	gamma = np.zeros([ne, nq, 1])
	LIB.get_specificheatratio(
		ctypes.c_void_p(Uq.ctypes.data), 
		ctypes.c_void_p(gamma.ctypes.data),
		ctypes.c_int(ne), 
		ctypes.c_int(nq), 
		ctypes.c_int(ns),
		ctypes.c_int(nsp),
		ctypes.c_int(physics.NDIMS),
		physics.c_cantera_file()
			)
	return gamma


def get_maxwavespeed(physics, Uq):

	gamma = get_specificheatratio(physics, Uq)
	P = get_pressure(physics, Uq)

	irho, irhou, irhoE, irhoYO2 = physics.get_state_indices()
	smom = physics.get_momentum_slice()

	rho = Uq[:, :, [irho]]
	mom = Uq[:, :, smom]

	return np.linalg.norm(mom, axis=2, keepdims=True) / rho \
			+ np.sqrt(gamma * P / rho)
