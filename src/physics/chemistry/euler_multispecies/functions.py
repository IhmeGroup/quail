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
from scipy.optimize import fsolve, root

from physics.base.data import (FcnBase, BCWeakRiemann, BCWeakPrescribed,
        SourceBase, ConvNumFluxBase)


class FcnType(Enum):
    SodMultispeciesAir = auto()


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

	def get_state(self, physics, x, t):
		# Unpack
		xshock = 0.0

		srho, srhou, srhoE, srhoYO2, srhoYN2 = physics.get_state_slices()

		#gamma = physics.gamma

		''' Pre-shock state '''
		rhoL = 1.0
		pL = 1.0
		uL = 0.

		''' Post-shock state '''
		rhoR = 0.125
		uR = 0.
		pR = 0.1

		''' Fill state '''
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		import code; code.interact(local=locals())

		for elem_ID in range(Uq.shape[0]):
			ileft = (x[elem_ID] < xshock).reshape(-1)
			iright = (x[elem_ID] >= xshock).reshape(-1)

			# Density
			Uq[elem_ID, iright, srho] = rhoR
			Uq[elem_ID, ileft, srho] = rhoL
			# Momentum
			Uq[elem_ID, iright, srhou] = rhoR*uR
			Uq[elem_ID, ileft, srhou] = rhoL*uL
			# Energy
			Uq[elem_ID, iright, srhoE] = pR/(gamma - 1.) + 0.5*rhoR*uR*uR
			Uq[elem_ID, ileft, srhoE] = pL/(gamma - 1.) + 0.5*rhoL*uL*uL

		return Uq # [ne, nq, ns]