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
#		File : src/defaultparams.py
#
#		Contains default input parameters.
#
# ------------------------------------------------------------------------ #

''' Restart parameters '''
Restart = {
	"File" : None,
		# If file name provided (str), then will restart from said data file
		# (pickle format)
	"StartFromFileTime" : True
		# If True, then will restart from time saved in restart file
}


''' Time stepping parameters '''
TimeStepping = {
	"InitialTime" : 0.,
		# Initial time
	"FinalTime" : None,
		# Final time
	"NumTimeSteps" : None,
		# Number of time steps (1st priority)
	"TimeStepSize" : None,
		# Time step size (2nd priority)
	"CFL" : None,
		# CFL number (3rd priority)
	"TimeStepper" : "RK4",
		# Time stepping scheme
		# See general.StepperType
	"OperatorSplittingExplicit" : "SSPRK3",
		# Explicit time stepping scheme for source terms if doing operator
		# splitting
		# See general.StepperType
	"OperatorSplittingImplicit" : "BDF1",
		# Implicit time stepping scheme for source terms if doing operator
		# splitting
		# See general.SourceStepperType
	"ODEScheme" : "FE",
		# Sets the specific time integration scheme when choosing to solve
		# an ODE or system of ODEs alone (see physics/zerodimensional
		# for examples)
}


''' Numerics parameters '''
Numerics = {
	"SolutionOrder" : 1,
		# Polynomial order of solution approximation
	"SolutionBasis" : "LagrangeSeg",
		# Basis type of solution approximation
		# See general.BasisType
	"Solver" : "DG",
		# Solver type
		# See general.SolverType
	"ElementQuadrature" : "GaussLegendre",
		# Quadrature type for integration over elements
		# See general.QuadratureType
	"FaceQuadrature" : "GaussLegendre",
		# Quadrature type for integration over faces
		# See general.QuadratureType
	"NodeType" : "Equidistant",
		# Node location type (for nodal basis functions)
		# See general.NodeType
	"ColocatedPoints" : False,
		# If True, quadrature points will be the solution nodes. This is
		# sometimes referred to as the DG spectral element method. If False,
		# overintegration is used, i.e. quadrature of high order, determined
		# by the physics type and the geometric order
	"L2InitialCondition" : True,
		# If True, will perform L2 projection to initialize the solution.
		# Otherwise, will interpolate to the nodes (only valid for a nodal
		# basis).
	"ApplyLimiters" : [],
		# Limiter type
		# By default, no limiter will be applied; otherwise, the name
		# of the limiter should be entered. A list of limiter name(s)
		# can also be used, allowing for multiple limiters.
		# See general.LimiterType
	"ShockIndicator" : None,
		# Shock Indicator Type
		# If None, then no shock indicator is applied
		# See general.ShockIndicatorType
	"TVBParameter" : 100.,
		# TVB MinMod Indicator
		# Default is set to 100.
		# This parameter modifies the sensitivity of the MinMod shock
		# indicator. It is problem dependent.
	"ArtificialViscosity" : False,
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	"AVParameter" : 1.,
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
	"SourceTreatmentADER" : "Explicit",
		# Treatment of source terms for ADER-DG
		# Either "Explicit" or "Implicit"
	"InterpolateFluxADER" : True,
		# If True, for ADER-DG, will obtain the flux by interpolating to
		# the nodes (only valid for a nodal basis). Otherwise, will perform
		# L2 projection
	"ConvFluxSwitch" : True,
		# If False, will ignore the convective flux
		# Useful for debugging
		# If ConvFluxSwitch is False, can easily test ODE solvers
	"SourceSwitch" : True,
		# If False, will ignore the source terms
		# Useful for debugging
	"PredictorGuessADER" : "Average",
		# How to construct initial guess for the ADER-DG scheme
		# Default is average value, can also select zeros or
		# ODEGuess
	"RecalculateJacobianADER" : False,
		# Either recalculates the source term jacobian or not for the
		# subiterations in the non-linear solver of the predictor step
	"PredictorThreshold" : 1e-15,
		# Sets the threshold requirement for the predictor step's
		# nonlinear solve. Lower values can be chosen which speeds up
		# the simulations, but at the cost of some error increase.
}


''' Mesh parameters '''
Mesh = {
	"File" : None,
		# Name of Gmsh mesh file to read
		# If None, then will create a uniform mesh on either a line segment
		# domain (1D) or a rectangular domain (2D) based on below parameters
	"ElementShape" : "Segment",
		# Shape of elements (if no Gmsh file provided)
	"NumElemsX" : 10,
		# Number of elements in the x-direction (if no Gmsh file provided)
	"NumElemsY" : 10,
		# Number of elements in the y-direction (if no Gmsh file provided)
	"xmin" : -1.,
		# x-coordinate of left boundary, "x1" (if no Gmsh file provided)
	"xmax" : 1.,
		# x-coordinate of right boundary, "x2" (if no Gmsh file provided)
	"ymin" : -1.,
		# y-coordinate of bottom boundary, "y1" (if no Gmsh file provided)
	"ymax" : 1.,
		# y-coordinate of top boundary, "y2" (if no Gmsh file provided)
	"PeriodicBoundariesX" : [],
		# List of the names of the two periodic boundaries in x-direction
		# If empty, then no periodicity in x-direction
	"PeriodicBoundariesY" : [],
		# List of the names of the two periodic boundaries in y-direction
		# If empty, then no periodicity in y-direction
}


''' Physics parameters '''
Physics = {
	"Type" : "ConstAdvScalar",
		# Physics type
		# See general.PhysicsType
	"Transport" : "NotNeeded",
		# Physics transport properties
		# See general.TransportType
	"ConvFluxNumerical" : "LaxFriedrichs",
		# Numerical convective flux
		# See ConvNumFluxType in functions.py in the corresponding physics
		# modules
	"DiffFluxNumerical" : None,
		# Numerical diffusive flux
		# See DiffNumFluxType in functions.py in the corresponding physics
		# modules
		
	# Physical parameters specific to the physics type are also set here.
	# Refer to the corresponding physics classes and the examples.
}


''' Initial condition parameters '''
InitialCondition = {
	"Function" : "Uniform",
		# Function to prescribe initial condition
		# See FcnType in functions.py in the corresponding physics
		# modules

	# Parameters specific to the Function are also set here.
	# Refer to the corresponding Function classes and the examples.
}

''' Exact solution parameters '''
ExactSolution = {
	# The purpose of the (optional) exact solution is for computing error.
	# If no keys and values are provided, then no exact solution will be
	# processed. If an exact solution is desired, the Function and associated
	# parameters should be set here. Refer to the corresponding Function
	# classes and the examples.
}


''' Boundary condition parameters '''
BoundaryConditions = {
	# A boundary condition must be set for each boundary. Keys and values
	# need not be provided only if the domain is fully periodic. See
	# BCType in functions.py in the corresponding physics modules. Refer to
	# the corresponding BC classes and the examples.
}


''' Source term parameters '''
SourceTerms = {
	# (Optional) source terms and associated parameters set here. See
	# SourceType in functions.py in the corresponding physics modules.
	# Refer to the corresponding source term classes and the examples.
}


''' Output parameters '''
Output = {
	"Prefix" : "Data",
		# Data files will have this prefix
	"WriteInterval" : -1,
		# Data files will be written at this interval
		# If nonpositive, then no files will be written
	"WriteInitialSolution" : False,
		# If True, then a data file will be written for the initial condition
	"WriteFinalSolution" : True,
		# If True, then a data file will be written for the final solution
	"AutoPostProcess" : True,
		# If True, then postprocessing script (if provided) will be
		# automatically called at the end of the simulation
	"Verbose" : False,
		# If False, minimal info will be printed to console, specifically
		# time step and residual information
		# If True, will also print out the input deck and the min/max of
		# the state variables
	"ProgressBar" : False,
		# If False, iteration info is printed to console
		# If True, a progress bar is given instead of iteration info
	"CustomFunctionFilename" : "custom_user_function"
		# Name of the user's custom function definitions.
}
