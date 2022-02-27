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
#       File : src/physics/euler/functions.py
#
#       Contains definitions of Functions, boundary conditions, and source
#       terms for the Euler equations.
#
# ------------------------------------------------------------------------ #
from enum import Enum, auto
import numpy as np
from scipy.optimize import fsolve, root

import errors
import general

from physics.base.data import (FcnBase, BCWeakRiemann, BCWeakPrescribed,
        SourceBase, ConvNumFluxBase)


class FcnType(Enum):
	'''
	Enum class that stores the types of analytical functions for initial
	conditions, exact solutions, and/or boundary conditions. These
	functions are specific to the available Euler equation sets.
	'''
	WildfireBurn = auto() 

# class BCType(Enum):
# 	'''
# 	Enum class that stores the types of boundary conditions. These
# 	boundary conditions are specific to the available Euler equation sets.
# 	'''
# 	SlipWall = auto()
# 	PressureOutlet = auto()
# 	ConstantTemp = auto()


class SourceType(Enum):
	'''
	Enum class that stores the types of source terms. These
	source terms are specific to the available Euler equation sets.
	'''
	WildfireSource = auto()


# class ConvNumFluxType(Enum):
# 	'''
# 	Enum class that stores the types of convective numerical fluxes. These
# 	numerical fluxes are specific to the available Euler equation sets.
# 	'''
# 	Roe = auto()


'''
---------------
State functions
---------------
These classes inherit from the FcnBase class. See FcnBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the FcnType enum members above.
'''

class WildfireBurn(FcnBase):
	'''
	Initial conditions of the wildfire 

	Attributes:
	-----------
	rho_woodi: float
		base density of wood
	rho_wateri: float
		base water density
	Tsi: float
		base temperature of the fuel 
	'''
	def __init__(self, rho_woodi=10., rho_wateri=1000., Tsi=298.):
		'''
		This method initializes the attributes.

		Inputs:
		-------
			rho_woodi: initial density of the fuel [kg/m^3]
			rho_wateri: initial density of the water [kg/m^3]
			Tsi: initial temperature of the fuel (K)

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.rho_woodi = rho_woodi
		self.rho_wateri = rho_wateri
		self.Tsi = Tsi

	def get_state(self, physics, x, t):
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		Nwood = physics.Nwood
		Cpwood = physics.Cpwood

		''' Base flow '''
		rho_woodi = self.rho_woodi
		rho_wateri = self.rho_wateri
		Tsi = self.Tsi 

		Uq[:, :, 0] = rho_woodi
		Uq[:, :, 1] = rho_wateri
		Uq[:, :, 2] = Tsi 


		return Uq # [ne, nq, ns]


'''
-------------------
Boundary conditions
-------------------
These classes inherit from either the BCWeakRiemann or BCWeakPrescribed
classes. See those parent classes for detailed comments of attributes
and methods. Information specific to the corresponding child classes can be
found below. These classes should correspond to the BCType enum members
above.
'''

# class ConstantTemp(BCWeakPrescribed):
# 	def __init__(self, Tsu=1.):
# 		self.Tsu = Tsu; 
# 	def get_boundary_state(self,physics,UqI,normals,x,t):
# 		UqB = UqI.copy() # get the interior state 

# 		UqB[:,:,Ts] = self.Tsu # set temperature of the boundary to the user specified value. (~400 K)

# 		return UqB 

'''
---------------------
Source term functions
---------------------
These classes inherit from the SourceBase class. See SourceBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the SourceType enum members above.
'''

class WildfireSource(SourceBase): 
	'''
	Source term for the Wildfire model (see above). Reference:
		[1] Linn, R., Reisner, J., Colman, J. J., & Winterkamp, J. (2002). 
		Studying wildfire behavior using FIRETEC. International journal of 
		wildland fire, 11(4), 233-246.
	'''
	
	def get_source(self, physics, Uq, x, t):
		Cpwood = physics.Cpwood
		Nwood = physics.Nwood
		Fwood = physics.Fwood 
		Fwater = physics.Fwater
		Z = physics.Z

		irhowood, irhowater, iTs = physics.get_state_indices()
		srhowood, srhowater, sTs = physics.get_state_slices()

		S = np.zeros_like(Uq)

		S[irhowood,:,:] = -Nwood*Fwood 
		S[:,irhowater,:] = -Fwater
		S[:,:,iTs] = Z+iTs

		return S

'''
------------------------
Numerical flux functions
------------------------
These classes inherit from the ConvNumFluxBase or DiffNumFluxBase class. 
See ConvNumFluxBase/DiffNumFluxBase for detailed comments of attributes 
and methods. Information specific to the corresponding child classes can 
be found below. These classes should correspond to the ConvNumFluxType 
or DiffNumFluxType enum members above.
'''