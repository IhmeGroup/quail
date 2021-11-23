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
#		File : src/general.py
#
#		Contains commonly used enums, constants, and settings.
#
# ------------------------------------------------------------------------ #
from enum import Enum, auto
import numpy as np

import defaultparams


'''
--------
Settings
--------
'''

np.set_printoptions(precision=15)


'''
---------
Constants
---------
'''

# "Zero"
eps = 1.e-15


'''
-------------
General enums
-------------
'''

class ShapeType(Enum):
	'''
	This enum contains the available shape types. See
	src/numerics/basis/basis.py for more information.
	'''
	Point = auto()
	Segment = auto()
	Quadrilateral = auto()
	Triangle = auto()
	Hexahedron = auto()


class BasisType(Enum):
	'''
	This enum contains the available basis types. See
	src/numerics/basis/basis.py for more information.
	'''
	LagrangeSeg = auto()
		# Lagrange basis polynomials for line segments
	LagrangeQuad = auto()
		# Lagrange basis polynomials for quadrilaterals
	LagrangeTri = auto()
		# Lagrange basis polynomials for triangles
	LagrangeHex = auto()
		# Lagrange basis polynomials for hexahedra
	LegendreSeg = auto()
		# Legendre basis polynomials for segments
	LegendreQuad = auto()
		# Legendre basis polynomials for quadrilaterals
	HierarchicH1Tri = auto()
		# Modal basis functions for triangles


class LimiterType(Enum):
	'''
	This enum contains the available limiter types. See
	src/numerics/limiting/ for more information.
	'''
	PositivityPreserving = auto()
	PositivityPreservingChem = auto()
	WENO = auto()

class ShockIndicatorType(Enum):
	'''
	This enum contains the available shoock indicator 
	types. See src/numerics/limiting/ for more info.
	'''
	MinMod = auto()
		# Uses a TVB modified minmod to tag troubled elements.
	

class SolverType(Enum):
	'''
	This enum contains the available solver types. See
	src/solver/ for more information.
	'''
	DG = auto()
		# Standard DG solver
	ADERDG = auto()
		# ADER-DG solver


class StepperType(Enum):
	'''
	This enum contains the available types of time stepping. See
	src/numerics/timestepping for more information.
	'''
	FE = auto()
		# Forward Euler
	RK4 = auto()
		# Standard 4th-order Runge-Kutta
	LSRK4 = auto()
		# Low-storage 4th-order Runge-Kutta
	SSPRK3 = auto()
		# Strong stability-preserving third-order Runge-Kutta
	ADER = auto()
		# ADER
	Strang = auto()
		# Strang splitting for reacting flows
	Simpler = auto()
		# Simpler scheme for reacting flows
	ODEIntegrator = auto()
		# Setup to only call for an ODE integrator (can use any steppers)

class SourceStepperType(Enum):
	'''
	This enum contains the available types of implicit time stepping for
	source terms in reacting flows. See src/numerics/timestepping for more
	information.
	'''
	BDF1 = auto()
		# 1st-order backwards differencing
	Trapezoidal = auto()
		# 2nd-order trapezoidal method
	LSODA = auto()
		# Scipy LSODA built-in ode solver


class PhysicsType(Enum):
	'''
	This enum contains the available physics types. See src/physics/
	for more information.
	'''
	ConstAdvScalar = auto()
		# Scalar advection with constant velocity (1D and 2D)
	ConstAdvDiffScalar = auto()
		# Scalar advection diffusion with constant velocity and diffusion
		# coefficients
	Burgers = auto()
		# Burgers equation (1D only)
	Euler = auto()
		# Euler equations (1D and 2D)
	NavierStokes = auto()
		# Navier-Stokes equations (1D and 2D)
	Chemistry = auto()
		# Euler equations with chemistry (1D and 2D)
	ModelProblem = auto()
		# Classic model problem for time convergence testing
	ModelPSRScalar = auto()
		# Model for perfectly stirred reactor (0D model)
	Pendulum = auto()
		# Model for 2nd order pendulum ODE (0D model)
	MultispeciesPSR = auto()
		# Model for PSR (with H2/Air chemistry)
		
		
class TransportType(Enum):
	'''
	This enum contains the available transport types for Navier-Stokes.
	see src/physics/navierstokes/tools.py for more information
	'''
	Constant = auto()
		# Constant transport properties
	Sutherland = auto()
		# Transport properties defined using sutherland's law for viscosity
	NotNeeded = auto()
		# No transport properties required for physics class

class ModalOrNodal(Enum):
	'''
	This enum contains flags indicating whether the basis functions are
	modal or nodal.
	'''
	Modal = auto()
		# Modal basis functions
	Nodal = auto()
		# Nodal basis functions
	Neither = auto()


class QuadratureType(Enum):
	'''
	This enum contains the available quadrature types.
	'''
	GaussLegendre = auto()
		# Gauss-Legendre quadrature
	GaussLobatto = auto()
		# Gauss-Lobatto quadrature (segments and quadrilaterals only)
	Dunavant = auto()
		# Dunavant quadrature (triangles only)


class NodeType(Enum):
	'''
	This enum contains the available solution node location types. only
	relevant for nodal basis functions.
	'''
	Equidistant = auto()
		# Equidistant nodes
	GaussLobatto = auto()
		# Gauss-Lobatto nodes (segments and quadrilaterals only)


INTERIORFACE = -1
NULLFACE = -2

solver_params = {**defaultparams.TimeStepping, **defaultparams.Numerics, **defaultparams.Output, **defaultparams.Restart}


def set_solver_params(params=None, **kwargs):
	if params is None:
		params = solver_params
		params["RestartFile"] = solver_params["File"]
	for key in kwargs:
		if key not in params.keys():
			raise KeyError
		params[key] = kwargs[key]
	return params


def pass_function(*args):
	pass


def zero_function(*args):
	return 0.