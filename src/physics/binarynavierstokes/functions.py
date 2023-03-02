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
#       File : src/physics/binarynavierstokes/functions.py
#
#       Contains definitions of Functions, boundary conditions, and source
#       terms for the 1D and 2D NS equations with a simple transport equation
#		for mass fraction.
#
# ------------------------------------------------------------------------ #
from enum import Enum, auto
import numpy as np
from scipy.optimize import fsolve, root

from physics.base.data import (FcnBase, BCWeakRiemann, BCWeakPrescribed,
        SourceBase, ConvNumFluxBase)


class FcnType(Enum):
	Waves1D = auto()
	Waves2D = auto()
	Waves2D2D = auto()

class SourceType(Enum):
	ManufacturedSourceBinary = auto()


class Waves1D(FcnBase):
	'''
	Manufactured solution to the Navier-Stokes equations used for
	verifying the order of accuracy of a given scheme.
	'''

	def __init__(self):
		pass

	def get_state(self, physics, x, t):
		p0 = 1e5
		T0 = 300.
		u0 = 10.
		Y0 = 0.5

		# Unpack
		gamma0 = physics.gamma0
		gamma1 = physics.gamma1
		R0 = physics.R0
		R1 = physics.R1
		mu0 = physics.mu0
		mu1 = physics.mu1
		Pr = physics.Pr
		Sc = physics.Sc
		cv0 = physics.cv0
		cv1 = physics.cv1

		irho, irhou, irhoE, irhoY = physics.get_state_indices()

		x1 = x[:, :, 0]

		''' Fill state '''
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		pressure = p0*(1.+0.1*np.sin(2.*np.pi*x1))
		temperature = T0*(1.+0.3*np.sin(2.*np.pi*x1))
		u = u0*(1.+0.3*np.sin(2.*np.pi*x1))
		Y = Y0*(1.+0.3*np.sin(2.*np.pi*x1))

		R = Y*R0 + (1.-Y)*R1
		cv = Y*cv0 + (1.-Y)*cv1

		rho = pressure / R / temperature
		rhou = rho * u
		rhoE = rho * (cv * temperature + 0.5 * u**2)
		rhoY = rho * Y

		Uq[:, :, irho] = rho

		Uq[:, :, irhou] = rhou

		Uq[:, :, irhoE] = rhoE

		Uq[:, :, irhoY] = rhoY

		return Uq  # [ne, nq, ns]

class Waves2D(FcnBase):
	'''
	Manufactured solution to the Navier-Stokes equations used for
	verifying the order of accuracy of a given scheme.
	'''

	def __init__(self):
		pass

	def get_state(self, physics, x, t):
		p0 = 1e5
		T0 = 300.
		u0 = 10.
		Y0 = 0.5

		# Unpack
		gamma0 = physics.gamma0
		gamma1 = physics.gamma1
		R0 = physics.R0
		R1 = physics.R1
		mu0 = physics.mu0
		mu1 = physics.mu1
		Pr = physics.Pr
		Sc = physics.Sc
		cv0 = physics.cv0
		cv1 = physics.cv1

		irho, irhou, irhov, irhoE, irhoY = physics.get_state_indices()

		x1 = x[:, :, 0]

		''' Fill state '''
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		pressure = p0*(1.+0.1*np.sin(2.*np.pi*x1))
		temperature = T0*(1.+0.3*np.sin(2.*np.pi*x1))
		u = u0*(1.+0.3*np.sin(2.*np.pi*x1))
		Y = Y0*(1.+0.3*np.sin(2.*np.pi*x1))

		R = Y*R0 + (1.-Y)*R1
		cv = Y*cv0 + (1.-Y)*cv1

		rho = pressure / R / temperature
		rhou = rho * u
		rhoE = rho * (cv * temperature + 0.5 * u**2)
		rhoY = rho * Y

		Uq[:, :, irho] = rho

		Uq[:, :, irhou] = rhou

		Uq[:, :, irhov] = 0.

		Uq[:, :, irhoE] = rhoE

		Uq[:, :, irhoY] = rhoY

		return Uq  # [ne, nq, ns]

class Waves2D2D(FcnBase):
	'''
	Manufactured solution to the Navier-Stokes equations used for
	verifying the order of accuracy of a given scheme.
	'''

	def __init__(self):
		pass

	def get_state(self, physics, x, t):
		p0 = 1e5
		T0 = 300.
		u0 = 10.
		Y0 = 0.5

		# Unpack
		gamma0 = physics.gamma0
		gamma1 = physics.gamma1
		R0 = physics.R0
		R1 = physics.R1
		mu0 = physics.mu0
		mu1 = physics.mu1
		Pr = physics.Pr
		Sc = physics.Sc
		cv0 = physics.cv0
		cv1 = physics.cv1

		irho, irhou, irhov, irhoE, irhoY = physics.get_state_indices()

		x1 = x[:, :, 0]
		x2 = x[:, :, 1]

		''' Fill state '''
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

		pressure = p0*(1.+0.1*np.sin(2.*np.pi*x1)+0.1*np.cos(2.*np.pi*x2)+0.1*np.cos(2.*np.pi*x1)*np.cos(2.*np.pi*x2))
		temperature = T0*(1.+0.1*np.sin(2.*np.pi*x1)+0.1*np.cos(2.*np.pi*x2)+0.1*np.cos(2.*np.pi*x1)*np.cos(2.*np.pi*x2))
		u = u0*(1.+0.2*np.sin(2.*np.pi*x1)+0.2*np.cos(2.*np.pi*x2)+0.2*np.cos(2.*np.pi*x1)*np.cos(2.*np.pi*x2))
		Y = Y0*(1.+0.1*np.sin(2.*np.pi*x1)+0.1*np.cos(2.*np.pi*x2)+0.1*np.cos(2.*np.pi*x1)*np.cos(2.*np.pi*x2))

		R = Y*R0 + (1.-Y)*R1
		cv = Y*cv0 + (1.-Y)*cv1

		rho = pressure / R / temperature
		rhou = rho * u
		rhoE = rho * (cv * temperature + 0.5 * u**2)
		rhoY = rho * Y

		Uq[:, :, irho] = rho

		Uq[:, :, irhou] = rhou

		Uq[:, :, irhov] = 0.

		Uq[:, :, irhoE] = rhoE

		Uq[:, :, irhoY] = rhoY

		return Uq  # [ne, nq, ns]


class ManufacturedSourceBinary(SourceBase):
	'''
	Generated source term for the Waves1D manufactured solution of the
	Navier-Stokes equations.
	'''

	def get_source(self, physics, Uq, x, t):
		# Unpack
		gamma0 = physics.gamma0
		gamma1 = physics.gamma1
		R0 = physics.R0
		R1 = physics.R1
		mu0 = physics.mu0
		mu1 = physics.mu1
		Pr = physics.Pr
		Sc = physics.Sc
		cv0 = physics.cv0
		cv1 = physics.cv1

		irho, irhou, irhoE, irhoY = physics.get_state_indices()
		x1 = x[:, :, 0]

		Sq = np.zeros_like(Uq)
		Sq[:,:,irho], Sq[:,:,irhou], Sq[:,:,irhoE], Sq[:,:,irhoY] = self.manufactured_source(x1, t, R0, R1, cv0, cv1, mu0, mu1, Pr, Sc)

		return Sq  # [ne, nq, ns]

	def manufactured_source(self, x1, t, R0, R1, cv0, cv1, mu0, mu1, Pr, Sc):
		S_rho = 20000.0 * np.pi * (3.0 * np.sin(2 * np.pi * x1) + 10.0) * np.cos(2 * np.pi * x1) / (
					(R0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
						90.0 * np.sin(2 * np.pi * x1) + 300.0)) + 6.0 * np.pi * (
							10000.0 * np.sin(2 * np.pi * x1) + 100000.0) * np.cos(2 * np.pi * x1) / ((R0 * (
					0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (90.0 * np.sin(
			2 * np.pi * x1) + 300.0)) - 0.002 * np.pi * (3.0 * np.sin(2 * np.pi * x1) + 10.0) * (
							10000.0 * np.sin(2 * np.pi * x1) + 100000.0) * np.cos(2 * np.pi * x1) / ((R0 * (
					0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (0.3 * np.sin(
			2 * np.pi * x1) + 1) ** 2) + (
							-0.3 * np.pi * R0 * np.cos(2 * np.pi * x1) + 0.3 * np.pi * R1 * np.cos(2 * np.pi * x1)) * (
							3.0 * np.sin(2 * np.pi * x1) + 10.0) * (10000.0 * np.sin(2 * np.pi * x1) + 100000.0) / ((
																																R0 * (
																																	0.15 * np.sin(
																																2 * np.pi * x1) + 0.5) + R1 * (
																																			0.5 - 0.15 * np.sin(
																																		2 * np.pi * x1))) ** 2 * (
																																90.0 * np.sin(
																															2 * np.pi * x1) + 300.0))

		S_rhou = 16.0 * np.pi ** 2 * (
					mu0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + mu1 * (0.5 - 0.15 * np.sin(2 * np.pi * x1))) * np.sin(
			2 * np.pi * x1) - 8.0 * np.pi * (0.3 * np.pi * mu0 * np.cos(2 * np.pi * x1) - 0.3 * np.pi * mu1 * np.cos(
			2 * np.pi * x1)) * np.cos(2 * np.pi * x1) + 20000.0 * np.pi * np.cos(2 * np.pi * x1) + 2000000.0 * np.pi * (
							 0.3 * np.sin(2 * np.pi * x1) + 1) ** 2 * np.cos(2 * np.pi * x1) / ((R0 * (
					0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (90.0 * np.sin(
			2 * np.pi * x1) + 300.0)) + 120.0 * np.pi * (0.3 * np.sin(2 * np.pi * x1) + 1) * (
							 10000.0 * np.sin(2 * np.pi * x1) + 100000.0) * np.cos(2 * np.pi * x1) / ((R0 * (
					0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (90.0 * np.sin(
			2 * np.pi * x1) + 300.0)) - 0.2 * np.pi * (10000.0 * np.sin(2 * np.pi * x1) + 100000.0) * np.cos(
			2 * np.pi * x1) / (R0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (
					0.5 - 0.15 * np.sin(2 * np.pi * x1))) + 100.0 * (
							 -0.3 * np.pi * R0 * np.cos(2 * np.pi * x1) + 0.3 * np.pi * R1 * np.cos(2 * np.pi * x1)) * (
							 0.3 * np.sin(2 * np.pi * x1) + 1) ** 2 * (10000.0 * np.sin(2 * np.pi * x1) + 100000.0) / ((
																																   R0 * (
																																	   0.15 * np.sin(
																																   2 * np.pi * x1) + 0.5) + R1 * (
																																			   0.5 - 0.15 * np.sin(
																																		   2 * np.pi * x1))) ** 2 * (
																																   90.0 * np.sin(
																															   2 * np.pi * x1) + 300.0))

		S_rhoE = 16.0 * np.pi ** 2 * (
					mu0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + mu1 * (0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
							 3.0 * np.sin(2 * np.pi * x1) + 10.0) * np.sin(2 * np.pi * x1) - 48.0 * np.pi ** 2 * (
							 mu0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + mu1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * np.cos(2 * np.pi * x1) ** 2 - 8.0 * np.pi * (
							 0.3 * np.pi * mu0 * np.cos(2 * np.pi * x1) - 0.3 * np.pi * mu1 * np.cos(
						 2 * np.pi * x1)) * (3.0 * np.sin(2 * np.pi * x1) + 10.0) * np.cos(2 * np.pi * x1) + (
							 3.0 * np.sin(2 * np.pi * x1) + 10.0) * (
							 20000.0 * np.pi * np.cos(2 * np.pi * x1) + 20000.0 * np.pi * ((cv0 * (
								 0.15 * np.sin(2 * np.pi * x1) + 0.5) + cv1 * (0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
																									   90.0 * np.sin(
																								   2 * np.pi * x1) + 300.0) + 50.0 * (
																									   0.3 * np.sin(
																								   2 * np.pi * x1) + 1) ** 2) * np.cos(
						 2 * np.pi * x1) / ((R0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
														90.0 * np.sin(2 * np.pi * x1) + 300.0)) - 0.002 * np.pi * ((
																															   cv0 * (
																																   0.15 * np.sin(
																															   2 * np.pi * x1) + 0.5) + cv1 * (
																																		   0.5 - 0.15 * np.sin(
																																	   2 * np.pi * x1))) * (
																															   90.0 * np.sin(
																														   2 * np.pi * x1) + 300.0) + 50.0 * (
																															   0.3 * np.sin(
																														   2 * np.pi * x1) + 1) ** 2) * (
										 10000.0 * np.sin(2 * np.pi * x1) + 100000.0) * np.cos(2 * np.pi * x1) / ((
																															  R0 * (
																																  0.15 * np.sin(
																															  2 * np.pi * x1) + 0.5) + R1 * (
																																		  0.5 - 0.15 * np.sin(
																																	  2 * np.pi * x1))) * (
																															  0.3 * np.sin(
																														  2 * np.pi * x1) + 1) ** 2) + (
										 10000.0 * np.sin(2 * np.pi * x1) + 100000.0) * (180.0 * np.pi * (
								 cv0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + cv1 * (
									 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * np.cos(2 * np.pi * x1) + (
																									 0.3 * np.pi * cv0 * np.cos(
																								 2 * np.pi * x1) - 0.3 * np.pi * cv1 * np.cos(
																								 2 * np.pi * x1)) * (
																									 90.0 * np.sin(
																								 2 * np.pi * x1) + 300.0) + 60.0 * np.pi * (
																									 0.3 * np.sin(
																								 2 * np.pi * x1) + 1) * np.cos(
						 2 * np.pi * x1)) / ((R0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (90.0 * np.sin(2 * np.pi * x1) + 300.0)) + ((
																																 cv0 * (
																																	 0.15 * np.sin(
																																 2 * np.pi * x1) + 0.5) + cv1 * (
																																			 0.5 - 0.15 * np.sin(
																																		 2 * np.pi * x1))) * (
																																 90.0 * np.sin(
																															 2 * np.pi * x1) + 300.0) + 50.0 * (
																																 0.3 * np.sin(
																															 2 * np.pi * x1) + 1) ** 2) * (
										 -0.3 * np.pi * R0 * np.cos(2 * np.pi * x1) + 0.3 * np.pi * R1 * np.cos(
									 2 * np.pi * x1)) * (10000.0 * np.sin(2 * np.pi * x1) + 100000.0) / ((R0 * (
								 0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (0.5 - 0.15 * np.sin(
						 2 * np.pi * x1))) ** 2 * (90.0 * np.sin(2 * np.pi * x1) + 300.0))) + 6.0 * np.pi * (
							 10000.0 * np.sin(2 * np.pi * x1) + 100000.0 + ((cv0 * (
								 0.15 * np.sin(2 * np.pi * x1) + 0.5) + cv1 * (0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
																						90.0 * np.sin(
																					2 * np.pi * x1) + 300.0) + 50.0 * (
																						0.3 * np.sin(
																					2 * np.pi * x1) + 1) ** 2) * (
										 10000.0 * np.sin(2 * np.pi * x1) + 100000.0) / ((R0 * (
								 0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
																									 90.0 * np.sin(
																								 2 * np.pi * x1) + 300.0))) * np.cos(
			2 * np.pi * x1) + 0.6 * np.pi ** 2 * (mu0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + mu1 * (
					0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (90.0 * np.sin(2 * np.pi * x1) + 300.0) * (
							 R0 - R1 + cv0 - cv1) * np.sin(2 * np.pi * x1) / Sc - 54.0 * np.pi ** 2 * (
							 mu0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + mu1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (R0 - R1 + cv0 - cv1) * np.cos(
			2 * np.pi * x1) ** 2 / Sc - 0.3 * np.pi * (
							 0.3 * np.pi * mu0 * np.cos(2 * np.pi * x1) - 0.3 * np.pi * mu1 * np.cos(
						 2 * np.pi * x1)) * (90.0 * np.sin(2 * np.pi * x1) + 300.0) * (R0 - R1 + cv0 - cv1) * np.cos(
			2 * np.pi * x1) / Sc + 360.0 * np.pi ** 2 * (R0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (
					0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (cv0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + cv1 * (
					0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (mu0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + mu1 * (
					0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (90.0 * np.sin(2 * np.pi * x1) + 300.0) * np.sin(
			2 * np.pi * x1) / (Pr * (10000.0 * np.sin(2 * np.pi * x1) + 100000.0)) - 32400.0 * np.pi ** 2 * (
							 R0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
							 cv0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + cv1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
							 mu0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + mu1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * np.cos(2 * np.pi * x1) ** 2 / (
							 Pr * (10000.0 * np.sin(2 * np.pi * x1) + 100000.0)) + 0.00036 * np.pi ** 2 * (
							 R0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
							 cv0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + cv1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
							 mu0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + mu1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
							 90.0 * np.sin(2 * np.pi * x1) + 300.0) * np.cos(2 * np.pi * x1) ** 2 / (
							 Pr * (0.1 * np.sin(2 * np.pi * x1) + 1) ** 2) - 180.0 * np.pi * (
							 R0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
							 cv0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + cv1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
							 0.3 * np.pi * mu0 * np.cos(2 * np.pi * x1) - 0.3 * np.pi * mu1 * np.cos(
						 2 * np.pi * x1)) * (90.0 * np.sin(2 * np.pi * x1) + 300.0) * np.cos(2 * np.pi * x1) / (
							 Pr * (10000.0 * np.sin(2 * np.pi * x1) + 100000.0)) - 180.0 * np.pi * (
							 R0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
							 mu0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + mu1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
							 0.3 * np.pi * cv0 * np.cos(2 * np.pi * x1) - 0.3 * np.pi * cv1 * np.cos(
						 2 * np.pi * x1)) * (90.0 * np.sin(2 * np.pi * x1) + 300.0) * np.cos(2 * np.pi * x1) / (
							 Pr * (10000.0 * np.sin(2 * np.pi * x1) + 100000.0)) - 180.0 * np.pi * (
							 cv0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + cv1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
							 mu0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + mu1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (
							 0.3 * np.pi * R0 * np.cos(2 * np.pi * x1) - 0.3 * np.pi * R1 * np.cos(2 * np.pi * x1)) * (
							 90.0 * np.sin(2 * np.pi * x1) + 300.0) * np.cos(2 * np.pi * x1) / (
							 Pr * (10000.0 * np.sin(2 * np.pi * x1) + 100000.0))

		S_rhoY = 20000.0 * np.pi * (0.15 * np.sin(2 * np.pi * x1) + 0.5) * (
					3.0 * np.sin(2 * np.pi * x1) + 10.0) * np.cos(2 * np.pi * x1) / ((R0 * (
					0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (90.0 * np.sin(
			2 * np.pi * x1) + 300.0)) + 6.0 * np.pi * (0.15 * np.sin(2 * np.pi * x1) + 0.5) * (
							 10000.0 * np.sin(2 * np.pi * x1) + 100000.0) * np.cos(2 * np.pi * x1) / ((R0 * (
					0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (90.0 * np.sin(
			2 * np.pi * x1) + 300.0)) - 0.002 * np.pi * (0.15 * np.sin(2 * np.pi * x1) + 0.5) * (
							 3.0 * np.sin(2 * np.pi * x1) + 10.0) * (
							 10000.0 * np.sin(2 * np.pi * x1) + 100000.0) * np.cos(2 * np.pi * x1) / ((R0 * (
					0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (0.3 * np.sin(
			2 * np.pi * x1) + 1) ** 2) + 0.3 * np.pi * (3.0 * np.sin(2 * np.pi * x1) + 10.0) * (
							 10000.0 * np.sin(2 * np.pi * x1) + 100000.0) * np.cos(2 * np.pi * x1) / ((R0 * (
					0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (0.5 - 0.15 * np.sin(2 * np.pi * x1))) * (90.0 * np.sin(
			2 * np.pi * x1) + 300.0)) + (
							 -0.3 * np.pi * R0 * np.cos(2 * np.pi * x1) + 0.3 * np.pi * R1 * np.cos(2 * np.pi * x1)) * (
							 0.15 * np.sin(2 * np.pi * x1) + 0.5) * (3.0 * np.sin(2 * np.pi * x1) + 10.0) * (
							 10000.0 * np.sin(2 * np.pi * x1) + 100000.0) / ((R0 * (
					0.15 * np.sin(2 * np.pi * x1) + 0.5) + R1 * (0.5 - 0.15 * np.sin(2 * np.pi * x1))) ** 2 * (
																						 90.0 * np.sin(
																					 2 * np.pi * x1) + 300.0)) + 0.6 * np.pi ** 2 * (
							 mu0 * (0.15 * np.sin(2 * np.pi * x1) + 0.5) + mu1 * (
								 0.5 - 0.15 * np.sin(2 * np.pi * x1))) * np.sin(2 * np.pi * x1) / Sc - 0.3 * np.pi * (
							 0.3 * np.pi * mu0 * np.cos(2 * np.pi * x1) - 0.3 * np.pi * mu1 * np.cos(
						 2 * np.pi * x1)) * np.cos(2 * np.pi * x1) / Sc

		return S_rho, S_rhou, S_rhoE, S_rhoY
