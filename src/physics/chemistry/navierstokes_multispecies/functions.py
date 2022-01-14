# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#		<https://github.com/IhmeGroup/quail>
#       
#		Copyright (C) 2020-2021
#
#       This program is distributed under the terms of the GNU
#		General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.  
#		If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#
#       File : src/physics/chemistry/navierstokes_multispecies/functions.py
#
#       Contains definitions of Functions, boundary conditions, and source
#       terms for the multispecies reacting Navier-Stokes equations
#
# ------------------------------------------------------------------------ #
from enum import Enum, auto
import numpy as np
import ctypes
from external.optional_cantera import ct

from scipy.optimize import fsolve, root

from physics.base.data import (FcnBase, BCWeakRiemann, BCWeakPrescribed,
        SourceBase, ConvNumFluxBase)
from external.optional_multispecies import multispecies_tools

import general

GAS_CONSTANT = 8.3144621000e3 # [J / (K kmole)]


class FcnType(Enum):
	DiffusionMixture = auto()


# class SourceType(Enum):
	# Reacting = auto()


# class ConvNumFluxType(Enum):
	# Roe = auto()


'''
State functions
'''
class DiffusionMixture(FcnBase):
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
		# Need to reset physical params since gas objects cant be saved
		# in pickle files
		if physics.gas is None:
			physics.gas = ct.Solution(physics.CANTERA_FILENAME)		

		srho, srhou, srhoE = physics.get_state_slices()

		# Set smooth varying function
		x10 = 25.e-3
		d = 2.5e-3
		f = 1. - np.exp(-1. * (x-x10)**2 / d**2)

		# Set the fuel and oxidizer states
		Tf = 320.0; To = 1350.0;
		YO2f = 0.195; YO2o = 0.142;
		YH2Of = 0.0; YH2Oo = 0.1;
		YCH4f = 0.214; YCH4o = 0.0;
		YN2f = 0.591; YN2o = 0.758;
		P = 101325.

		Yf = np.array([YCH4f, YO2f, YH2Of, YN2f])
		Yo = np.array([YCH4o, YO2o, YH2Oo, YN2o])

		''' Fill state '''
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		for elem_ID in range(Uq.shape[0]):
			for iq in range(Uq.shape[1]):

				T = To + (Tf - To) * f[elem_ID, iq, 0]
				Y = Yo + (Yf - Yo) * f[elem_ID, iq, 0]

				physics.gas.TPY = T, P, Y
				rho = physics.gas.TD[1]
				e = physics.gas.UV[0]

				# Density
				Uq[elem_ID, iq, srho] = rho
				# Momentum
				Uq[elem_ID, iq, srhou] = 0.0
				# Energy
				Uq[elem_ID, iq, srhoE] = rho * e # KE is zero
				# rhoY
				for isp in range(physics.NUM_SPECIES - 1):
					Uq[elem_ID, iq, 3 + isp] = rho * Y[isp]

		return Uq # [ne, nq, ns]