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
#       File : src/physics/navierstokes/tools.py
#
#       Contains tools for the navierstokes class that define the 
#		transport properties for the given setup
#
# ------------------------------------------------------------------------ #
import numpy as np
from general import TransportType


def set_transport(transport_type):
	'''
	Given the Transport parameter, set the get_transport function

	Inputs:
	-------
		transport_type: string to determine transport type

	Outputs:
	--------
		fcn: name of function to be passed
	'''
	
	if TransportType[transport_type] == TransportType.Constant:
		fcn = get_constant_transport
	elif TransportType[transport_type] == TransportType.Sutherland:
		fcn = get_sutherland_transport
	elif TransportType[transport_type] == TransportType.NotNeeded:
		fcn = None
	else:
		raise NotImplementedError("Transport not supported")

	return fcn


def get_constant_transport(physics, Uq, flag_non_physical=None):
	'''
	Returns viscosity and thermal conductivity for constant transport
	properties.

	Inputs:
	-------
		physics: physics object
		Uq: solution state evaluated at quadrature points [ne, nq, ns]
		flag_non_physical: boolean to check for physicality of transport

	Outputs:
	--------
		mu: viscosity [ne]
		kappa: thermal conductivity [ne]
	'''
	# Unpack
	Pr = physics.Pr
	gamma = physics.gamma
	R = physics.R

	mu = physics.mu0 * np.ones([Uq.shape[0], Uq.shape[1]])
	cv = 1./(gamma - 1) * R

	return mu, mu * cv * gamma / Pr


def get_sutherland_transport(physics, Uq, flag_non_physical=None):
	'''
	Calculates the viscosity and thermal conductivity using Sutherland's 
	law.

	Inputs:
	-------
		physics: physics object
		Uq: solution state evaluated at quadrature points [ne, nq, ns]
		flag_non_physical: boolean to check for physicality of transport

	Outputs:
	--------
		mu: viscosity [ne]
		kappa: thermal conductivity [ne]
	'''
	# Unpack
	Pr = physics.Pr
	gamma = physics.gamma
	R = physics.R
	s = physics.s
	T0 = physics.T0
	beta = physics.beta

	T = physics.compute_variable("Temperature",
			Uq, flag_non_physical=flag_non_physical, x=None)

	mu0 = 0.1; s = 1.; T0 = 1.; beta = 1.5;
	cv = 1./(gamma - 1) * R

	mu = mu0 * (T / T0)**beta * ((T0 + s) / (T + s))
	kappa = mu * cv * gamma / Pr

	return mu, kappa
