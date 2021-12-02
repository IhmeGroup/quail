# ------------------------------------------------------------------------ #
#
#       File : src/physics/chemistry/functions.py
#
#       Contains definitions of Functions, boundary conditions, and source
#       terms for the 1D Euler equations with a simple transport equation
#		for mass fraction.
#
# ------------------------------------------------------------------------ #
from enum import Enum, auto
import numpy as np
from external.optional_cantera import ct

from scipy.optimize import fsolve, root

from physics.base.data import (FcnBase, BCWeakRiemann, BCWeakPrescribed,
        SourceBase, ConvNumFluxBase)
from external.optional_thermo import thermo_tools


class FcnType(Enum):
    SodMultispeciesAir = auto()
    SodMultispeciesAir1sp = auto()


# class SourceType(Enum):
    # Arrhenius = auto()
    # Heaviside = auto()


# class ConvNumFluxType(Enum):
	# Roe = auto()


'''
State functions
'''
class SodMultispeciesAir(FcnBase):
	'''
	Attributes:
	-----------
	xshock: float
		initial location of shock
	'''
	def __init__(self):
		'''
		This method initializes the attributes.

		Inputs:
		-------
		    xshock: initial location of shock

		Outputs:
		--------
		    self: attributes initialized
		'''
	def set_state_from_primitives(self, physics, rho, P, u, Y):
		
		gas = physics.gas

		U = np.zeros([physics.NUM_STATE_VARS])
		irho, irhou, irhoE = physics.get_state_indices()

		U[irho] = rho
		U[irhou] = rho*u

		U[irhoE+1:] = rho * Y[:-1]

		W = thermo_tools.get_W_from_Y(gas.molecular_weights, Y)
		T = thermo_tools.get_T_from_rhop(rho, P, W)
		gas.TPY = T, P, "O2:{},N2:{}".format(Y[0, 0], Y[1, 0])

		gamma = thermo_tools.get_gamma(gas.cv, W)
		
		# Double check this definition
		U[irhoE] = rho * gas.UV[0] + 0.5*rho*u*u

		return U

	def get_state(self, physics, x, t):
		# Need to reset physical params since gas objects cant be saved
		# in pickle files
		if physics.gas is None:
			physics.gas = ct.Solution(physics.CANTERA_FILENAME)		

		xshock = 0.0

		srho, srhou, srhoE = physics.get_state_slices()

		#gamma = physics.gamma
		''' Pre-shock state '''
		rhoL = 1.0
		pL = 1.0*1e5
		uL = 0.
		Y = np.array([[0.21], [0.79]])

		''' Post-shock state '''
		rhoR = 0.125
		uR = 0.
		pR = 0.1*1e5
		# rhoR = 1.0
		# pR = 1.0
		# uR = 0.
		''' Fill state '''
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		# Set the left state
		stateL = self.set_state_from_primitives(physics, rhoL, pL, uL, Y)
		stateR = self.set_state_from_primitives(physics, rhoR, pR, uR, Y)

		for elem_ID in range(Uq.shape[0]):
			ileft = (x[elem_ID] < xshock).reshape(-1)
			iright = (x[elem_ID] >= xshock).reshape(-1)

			# Density
			Uq[elem_ID, iright, srho] = stateR[srho]
			Uq[elem_ID, ileft, srho] = stateL[srho]
			# Momentum
			Uq[elem_ID, iright, srhou] = stateR[srhou]
			Uq[elem_ID, ileft, srhou] = stateL[srhou]
			# Energy
			Uq[elem_ID, iright, srhoE] = stateR[srhoE]
			Uq[elem_ID, ileft, srhoE] = stateL[srhoE]
			# YO2
			Uq[elem_ID, iright, -1] = stateR[-1]
			Uq[elem_ID, ileft, -1] = stateL[-1]	
			# YN2
			# Uq[elem_ID, iright, srhoYN2] = stateR[srhoYN2]
			# Uq[elem_ID, ileft, srhoYN2] = stateL[srhoYN2]		

		return Uq # [ne, nq, ns]

class SodMultispeciesAir1sp(FcnBase):
	'''
	Attributes:
	-----------
	xshock: float
		initial location of shock
	'''
	def __init__(self):
		'''
		This method initializes the attributes.

		Inputs:
		-------
		    xshock: initial location of shock

		Outputs:
		--------
		    self: attributes initialized
		'''

	def set_state_from_primitives(self, physics, rho, P, u, Y):
		
		gas = physics.gas

		U = np.zeros([physics.NUM_STATE_VARS])
		irho, irhou, irhoE = physics.get_state_indices()

		U[irho] = rho
		U[irhou] = rho*u

		U[irhoE+1:] = rho * Y[0]

		W = thermo_tools.get_W_from_Y(gas.molecular_weights, Y)
		T = thermo_tools.get_T_from_rhop(rho, P, W)
		gas.TPY = T, P, "N2:{}".format(Y[0, 0])

		gamma = thermo_tools.get_gamma(gas.cv, W)
		
		# Double check this definition
		U[irhoE] = rho * gas.UV[0] + 0.5*rho*u*u

		return U

	def get_state(self, physics, x, t):
		# Need to reset physical params since gas objects cant be saved
		# in pickle files
		if physics.gas is None:
			physics.gas = ct.Solution(physics.CANTERA_FILENAME)		

		xshock = 0.0

		srho, srhou, srhoE = physics.get_state_slices()

		#gamma = physics.gamma
		''' Pre-shock state '''
		rhoL = 1.0
		pL = 1.0*1e5
		uL = 0.
		Y = np.array([[1.0]])

		''' Post-shock state '''
		rhoR = 0.125
		uR = 0.
		pR = 0.1*1e5
		# rhoR = 1.0
		# pR = 1.0
		# uR = 0.
		''' Fill state '''
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		# Set the left state
		stateL = self.set_state_from_primitives(physics, rhoL, pL, uL, Y)
		stateR = self.set_state_from_primitives(physics, rhoR, pR, uR, Y)

		for elem_ID in range(Uq.shape[0]):
			ileft = (x[elem_ID] < xshock).reshape(-1)
			iright = (x[elem_ID] >= xshock).reshape(-1)

			# Density
			Uq[elem_ID, iright, srho] = stateR[srho]
			Uq[elem_ID, ileft, srho] = stateL[srho]
			# Momentum
			Uq[elem_ID, iright, srhou] = stateR[srhou]
			Uq[elem_ID, ileft, srhou] = stateL[srhou]
			# Energy
			Uq[elem_ID, iright, srhoE] = stateR[srhoE]
			Uq[elem_ID, ileft, srhoE] = stateL[srhoE]
			# YO2
			# Uq[elem_ID, iright, srhoYO2] = stateR[srhoYO2]
			# Uq[elem_ID, ileft, srhoYO2] = stateL[srhoYO2]	
			# YN2
			# Uq[elem_ID, iright, srhoYN2] = stateR[srhoYN2]
			# Uq[elem_ID, ileft, srhoYN2] = stateL[srhoYN2]		

		return Uq # [ne, nq, ns]