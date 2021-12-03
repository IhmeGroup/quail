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
#       File : src/solver/base.py
#
#       Contains class definitions for the solver base class
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import importlib
import numpy as np
import time

import errors

from general import ModalOrNodal, NodeType, ShapeType, QuadratureType, \
		StepperType, LimiterType, BasisType

import meshing.meshbase as mesh_defs
import meshing.tools as mesh_tools

import numerics.basis.tools as basis_tools

import numerics.helpers.helpers as helpers

import numerics.limiting.tools as limiter_tools

import numerics.timestepping.tools as stepper_tools
import numerics.timestepping.stepper as stepper_defs
import numerics.quadrature.segment as segment

import processing.post as post_defs
import processing.readwritedatafiles as readwritedatafiles

import solver.tools as solver_tools


class SolverBase(ABC):
	'''
	This is a base class for any solver type.

	Attributes:
	-----------
	params: dictionary
		contains a list of parameters that can be accessed with kwargs
	physics: object
		contains the set of equations to be solved
	mesh: object
		contains the geometric information for the solver's mesh
	time: float
		global time of the solution at the given time step
	itime: int
		global iteration count
	basis: object
		contains all the information and methods for the basis class
	order: int
		order of solution approximation
	state_coeffs: numpy array
		coefficients of polynomial approximation of global solution
	limiter: object
		contains all the information and methods for the limiter class
	verbose: bool
		determines whether to print detailed info to console
	min_state: numpy array
		minimum values of state variables
	max_state: numpy array
		maximum values of state variables

	Abstract Methods:
	-----------------
	precompute_matrix_helpers
		precomputes a variety of functions and methods prior to running the
		simulation
	get_element_residual
		calculates the residual contribution for elements
	get_interior_face_residual
		calculates the residual contribution for interior faces
	get_boundary_face_residual
		calculates the residual contribution for boundary faces

	Methods:
	--------
	check_compatibility
		checks parameter compatibilities based on the given input deck
	init_state_from_fcn
		initializes state from a specified function in the input deck
	project_state_to_new_basis
		takes a state from a restartfile and projects it onto a higher
		order of accuracy
	get_residual
		computes the full residual
	get_element_residuals
		computes the residual contributions from interiors of all elements
	get_interior_face_residuals
		computes the residual contributions from all interior faces
	get_boundary_face_residuals
		computes the residual contributions from all boundary faces
	apply_limiter
		applies limiter to the state
	get_min_max_state
		computes minimum and maximum values of the state
	print_info
		prints key info (at each time step)
	solve
		applies time integration scheme to solve the system
	'''
	def __init__(self, params, physics, mesh):
		self.params = params
		self.physics = physics
		self.mesh = mesh

		self.time = params["InitialTime"]

		# Set solution basis and order
		self.order = params["SolutionOrder"]
		basis_type  = params["SolutionBasis"]
		self.basis = basis_tools.set_basis(self.order, basis_type)
		# State polynomial coefficients (what we're solving for)
		self.state_coeffs = np.zeros([mesh.num_elems,
				self.basis.get_num_basis_coeff(self.order),
				physics.NUM_STATE_VARS])

		# Node type
		node_type = params["NodeType"]
		self.basis.get_1d_nodes = basis_tools.set_1D_node_calc(node_type)

		# Set quadrature
		self.basis.set_elem_quadrature_type(params["ElementQuadrature"])
		self.basis.set_face_quadrature_type(params["FaceQuadrature"])
		mesh.gbasis.set_elem_quadrature_type(params["ElementQuadrature"])
		mesh.gbasis.set_face_quadrature_type(params["FaceQuadrature"])

		self.basis.force_colocated_nodes_quad_pts(params["ColocatedPoints"])

		# Limiter
		limiter_types = params["ApplyLimiters"]
		shock_indicator_type = params["ShockIndicator"]
		tvb_param = params["TVBParameter"]
		# Cast to list
		if not isinstance(limiter_types, list):
			limiter_types = [limiter_types]
			params["ApplyLimiters"] = limiter_types
		self.limiters = []
		for limiter_type in limiter_types:
			limiter = limiter_tools.set_limiter(limiter_type,
					physics.PHYSICS_TYPE)
			if limiter:
				# Set shock indicator
				limiter_tools.set_shock_indicator(limiter,
						shock_indicator_type)
				# Set TVB Parameter
				limiter.tvb_param = tvb_param
				self.limiters.append(limiter)

		# Console output
		self.verbose = params["Verbose"]
		self.progress_bar = params["ProgressBar"]
		self.min_state = np.zeros(physics.NUM_STATE_VARS)
		self.max_state = np.zeros(physics.NUM_STATE_VARS)


		# Search for custom_user_function in case directory
		custom_user_function = self.params["CustomFunctionFilename"]
		try:
			user_fcn = importlib.import_module(custom_user_function)
			self.custom_user_function = \
					user_fcn.custom_user_function
		except ModuleNotFoundError:
			pass # Not an error, just pass as the user provided
				 # custom function file is not in the current
				 # directory.

		# Counter to compare ODE evaluations in ADERDG and Splitting methods
		self.count_evaluations = 0

		# Compatibility checks
		self.check_compatibility()

	def __repr__(self):
		return '{self.__class__.__name__}(Physics: {self.physics},\n   \
		Basis: {self.basis},\n   Stepper: {self.stepper})'.format(self=self)

	def check_compatibility(self):
		'''
		Checks a variety of parameter combinations to ensure compatibility
		for the specified input deck and the following simulation
		'''
		mesh = self.mesh
		params = self.params
		basis = self.basis

		# Check for same shape between mesh and solution
		if mesh.gbasis.SHAPE_TYPE != basis.SHAPE_TYPE:
			raise errors.IncompatibleError

		if not params["L2InitialCondition"] and basis.MODAL_OR_NODAL \
				!= ModalOrNodal.Nodal:
			raise errors.IncompatibleError

		# Gauss Lobatto nodes compatibility checks
		node_type = params["NodeType"]
		colocated_points = params["ColocatedPoints"]
		elem_quad = params["ElementQuadrature"]
		face_quad = params["FaceQuadrature"]

		# Compatibility check for GLL nodes with triangles
		if NodeType[node_type] == NodeType.GaussLobatto and \
			basis.SHAPE_TYPE == ShapeType.Triangle:
			raise errors.IncompatibleError

		# Compatibility check for colocated scheme, i.e. quadrature points
		# equal solution nodes
		if NodeType[node_type] == NodeType.Equidistant and colocated_points:
			raise errors.IncompatibleError
		if (QuadratureType[elem_quad] != QuadratureType.GaussLobatto or \
				QuadratureType[face_quad] != QuadratureType.GaussLobatto) \
				and colocated_points:
			raise errors.IncompatibleError

		# Colocated points only compatible with nodal basis
		if basis.MODAL_OR_NODAL is ModalOrNodal.Modal and colocated_points:
			raise errors.IncompatibleError

		# Check if ConvFluxSwitch or SourceSwitch are being used with
		# Strang/Simpler splitting
		source_switch = params["SourceSwitch"]
		convflux_switch = params["ConvFluxSwitch"]
		stepper_type = params["TimeStepper"]
		if not source_switch and \
				( StepperType[stepper_type] == StepperType.Strang or \
				  StepperType[stepper_type] == StepperType.Simpler ) :
			raise errors.IncompatibleError
		if not convflux_switch and \
				( StepperType[stepper_type] == StepperType.Strang or \
				  StepperType[stepper_type] == StepperType.Simpler ) :
			raise errors.IncompatibleError

		# Currently, positivity-preserving limiter not compatible with
		# modal triangular basis
		if LimiterType.PositivityPreserving.name in params["ApplyLimiters"] \
				or LimiterType.PositivityPreservingChem.name in params[
				"ApplyLimiters"]:
			if basis.BASIS_TYPE == BasisType.HierarchicH1Tri:
				raise errors.IncompatibleError

	@abstractmethod
	def precompute_matrix_helpers(self):
		'''
		Precomputes element and face helper arrays and structures that only
		need to be computed once at the beginning of the simulation.
		'''
		pass

	@abstractmethod
	def get_element_residual(self, Uc, res_elem):
		'''
		Calculates the volume contribution to the residual for all elements.

		Inputs:
		-------
			Uc: solution array for all elements (polynomial coefficients)
			res_elem: residual array for all elements

		Outputs:
		--------
			res_elem: calculated residual array for all elements
		'''
		pass

	@abstractmethod
	def get_interior_face_residual(self, faceL_IDs, faceR_IDs, UcL, UcR):
		'''
		Calculates the surface integral for the interior faces, divided
		between left and right contributions.

		Inputs:
		-------
			faceL_IDs: face IDs for each interior face from the perspective
				of each left neighboring element
			faceR_IDs: face IDs for each interior face from the perspective
				of each right neighboring element
			UcL: solution array for left neighboring element (polynomial
				coefficients)
			UcR: solution array for right neighboring element (polynomial
				coefficients)

		Outputs:
		--------
			resL: calculated residual array (left neighboring element
			contribution)
			resR: calculated residual array (right neighboring element
			contribution)
		'''
		pass

	@abstractmethod
	def get_boundary_face_residual(self, bgroup, face_IDs, Uc, resB):
		'''
		Calculates the residual from the surface integral for all boundary
		faces within a boundary group.

		Inputs:
		-------
			bgroup: boundary group object
			face_IDs: IDs of boundary faces
			Uc: solution array from adjacent element
			resB: residual array (for adjacent element)

		Outputs:
		--------
			resB: calculated residual array (from boundary face)
		'''
		pass


	def custom_user_function(self, solver):
		'''
		Placeholder for the custom_user_function. Users can specify the
		custom_user_function in an additional file. This would then be
		called each iteration.
		'''
		pass

	def init_state_from_fcn(self):
		'''
		Initializes the state (initial condition) from the specified
		function in the input deck. Either interpolates the state to the
		nodes or employs L2 projection to initialize the state.
		'''
		# Unpack
		mesh = self.mesh
		physics = self.physics
		basis = self.basis
		params = self.params
		iMM_elems = self.elem_helpers.iMM_elems

		U = self.state_coeffs
		ns = physics.NUM_STATE_VARS
		order = self.order

		# Get solution nodes or quadrature info
		if not params["L2InitialCondition"]:
			# Solution nodes
			eval_pts = basis.get_nodes(order)
		else:
			# Quadrature

			# Over-integrate
			order = 2*np.amax([self.order, 1])
			order = physics.get_quadrature_order(order)

			quad_order = basis.get_quadrature_order(mesh, order)
			quad_pts, quad_wts = basis.get_quadrature_data(quad_order)

			eval_pts = quad_pts
			nq = eval_pts.shape[0]

		# Compute state
		xphys = np.empty((mesh.num_elems,) + eval_pts.shape)

		for elem_ID in range(mesh.num_elems):
			xphys[elem_ID] = mesh_tools.ref_to_phys(mesh, elem_ID, eval_pts)
		f = physics.IC.get_state(physics, x=xphys, t=self.time)

		if not params["L2InitialCondition"]:
			# Interpolate to solution nodes
			solver_tools.interpolate_to_nodes(f, U)
		else:
			# L2 projection
			solver_tools.L2_projection(mesh, iMM_elems, basis, quad_pts,
					quad_wts, f, U)

	def project_state_to_new_basis(self, U_old, basis_old, order_old):
		'''
		Projects the state to a different basis and/or order

		Inputs:
		-------
			U_old: restart files old solution array
			basis_old: previous basis function
			order_old: previous polynomial order

		Outputs:
		--------
			state is modified
		'''
		mesh = self.mesh
		physics = self.physics
		basis = self.basis
		params = self.params
		iMM_elems = self.elem_helpers.iMM_elems

		U = self.state_coeffs
		ns = physics.NUM_STATE_VARS

		if basis_old.SHAPE_TYPE != basis.SHAPE_TYPE:
			raise errors.IncompatibleError

		if not params["L2InitialCondition"]:
			# Interpolate to solution nodes
			eval_pts = basis.get_nodes(self.order)
		else:
			# Quadrature
			order = 2*np.amax([self.order, order_old])
			quad_order = basis.get_quadrature_order(mesh, order)

			quad_pts, quad_wts = basis.get_quadrature_data(quad_order)
			eval_pts = quad_pts

		basis_old.get_basis_val_grads(eval_pts, get_val=True)

		Uq_old = helpers.evaluate_state(U_old, basis_old.basis_val)

		if not params["L2InitialCondition"]:
			solver_tools.interpolate_to_nodes(Uq_old, U)
		else:
			solver_tools.L2_projection(mesh, iMM_elems, basis, quad_pts,
					quad_wts, Uq_old, U)

	def get_residual(self, U, res):
		'''
		Calculates the surface + volume integral for the DG formulation

		Inputs:
		-------
			U: solution array

		Outputs:
		--------
			res: residual array
		'''
		mesh = self.mesh
		physics = self.physics
		stepper = self.stepper

		# Initialize residual to zero
		if stepper.balance_const is None:
			res[:] = 0.
		else:
			res[:] = stepper.balance_const

		self.get_boundary_face_residuals(U, res)
		self.get_element_residuals(U, res)
		self.get_interior_face_residuals(U, res)

		return res

	def get_element_residuals(self, U, res):
		'''
		Wrapper for get_element_residual (just for consistency with how
		interior/boundary face contributions are computed).

		Inputs:
		-------
			U: solution array

		Outputs:
		--------
			res: calculated residual array
		'''

		res = self.get_element_residual(U, res)

	def get_interior_face_residuals(self, U, res):
		'''
		Computes interior face residual contributions.

		Inputs:
		-------
			U: solution array
			res: residual array

		Outputs:
		--------
			res: calculated residual array (includes all interior face
				contributions)
		'''
		mesh = self.mesh
		int_face_helpers = self.int_face_helpers
		elemL_IDs = int_face_helpers.elemL_IDs
		elemR_IDs = int_face_helpers.elemR_IDs
		faceL_IDs = int_face_helpers.faceL_IDs
		faceR_IDs = int_face_helpers.faceR_IDs

		# Extract state coefficients of elements to the left and right of
		# this interior face
		UL = U[elemL_IDs]
		UR = U[elemR_IDs]

		# Calculate face residuals for left and right elements
		RL, RR, RL_diff, RR_diff = self.get_interior_face_residual(faceL_IDs, faceR_IDs, UL,
				UR)

		# Add this residual back to the global. The np.add.at function is
		# used to correctly handle duplicate element IDs.
		np.add.at(res, elemL_IDs, -RL)
		np.add.at(res, elemR_IDs,  RR)

		# Add the additional diffusion portion of the residual to the
		# correct left/right states.
		np.add.at(res, elemL_IDs,  RL_diff)
		np.add.at(res, elemR_IDs,  RR_diff)

	def get_boundary_face_residuals(self, U, res):
		'''
		Computes interior face residual contributions for all boundary
		groups.

		Inputs:
		-------
			U: solution array
			res: residual array

		Outputs:
		--------
			res: calculated residual array (includes all boundary face
				contributions)
		'''
		mesh = self.mesh
		physics = self.physics
		bface_helpers = self.bface_helpers
		elem_IDs = bface_helpers.elem_IDs
		face_IDs = bface_helpers.face_IDs

		# Loop through boundary groups
		for bgroup in mesh.boundary_groups.values():

			bgroup_elem_IDs = elem_IDs[bgroup.number]
			bgroup_face_IDs = face_IDs[bgroup.number]

			resB = self.get_boundary_face_residual(bgroup,
					bgroup_face_IDs, U[bgroup_elem_IDs],
					res[bgroup_elem_IDs])

			np.add.at(res, bgroup_elem_IDs, -resB)

	def apply_limiter(self, U):
		'''
		Applies the limiter to the solution array, U.

		Inputs:
		-------
			U: solution array

		Outputs:
		--------
			U: limited solution array
		'''
		for limiter in self.limiters:
			if limiter is not None:
				limiter.limit_solution(self, U)

	def get_min_max_state(self, Uq):
		'''
		Gets min and max values of state variables.

		Inputs:
		-------
			Uq: state variables evaluated at quadrature points [nq, ns]

		Outputs:
		--------
			self.min_state: minimum values of state variables
			self.max_state: maximum values of state variables
		'''
		self.min_state = np.minimum(self.min_state, np.amin(np.amin(Uq,
				axis=1), axis=0))
		self.max_state = np.maximum(self.max_state, np.amax(np.amax(Uq,
				axis=1), axis=0))

	def print_info(self, physics, res, itime, t, dt):
		'''
		Prints key information to console. If self.verbose is False, then
		only time and residual info is printed; otherwise, min and max
		values of the state are also reported. If self.progress_bar is True,
		then the iteration output is replaced with a progress bar.

		Inputs:
		-------
			physics: physics object
			res: residual array [num_elems, nb, ns]
			itime: time iteration
			t: time
			dt: time step size
		'''
		# Progress bar output
		if self.progress_bar:
			solver_tools.update_progress(t / self.stepper.tfinal)

		# Basic info: time, residual. If using a progress bar, only the last
		# iteration is output.
		is_final_iteration = itime == self.stepper.num_time_steps - 1
		if not self.progress_bar or is_final_iteration:
			print("%d: Time = %g - Time step = %g - Residual norm = %g" % (
					itime + 1, t, dt, np.linalg.norm(np.reshape(res, -1),
					ord=1)))

			# If requested, report min and max of state variables
			if self.verbose:
				print("\nMin|Max at volume quadrature points:")
				s = 0
				for state_var in physics.StateVariables:
					string = "    " + state_var.name + ": " + "%g | %g"
					print(string % (self.min_state[s], self.max_state[s]))
					s += 1

			print("------------------------------------------------------" + \
					"-------------------------")


	def solve(self):
		'''
		Performs the main solve of the DG method. Initializes the temporal
		loop.
		'''
		physics = self.physics
		mesh = self.mesh
		stepper = self.stepper
		t = self.time

		# Parameters for writing data
		write_interval = self.params["WriteInterval"]
		if write_interval == -1:
			write_interval = np.NAN
		write_final_solution = self.params["WriteFinalSolution"]
		write_initial_solution = self.params["WriteInitialSolution"]

		if write_initial_solution:
			readwritedatafiles.write_data_file(self, 0)

		t0 = time.time()

		print("\n\nUNSTEADY SOLVE:")
		print("--------------------------------------------------------" + \
				"-----------------------")

		# Custom user function initial iteration
		self.custom_user_function(self)

		solver.itime = 0
		while solver.itime < stepper.num_time_steps:
			# Reset min and max state
			self.max_state[:] = -np.inf
			self.min_state[:] = np.inf

			# Get time step size
			stepper.dt = stepper.get_time_step(stepper, self)

			# Integrate in time
			res = stepper.take_time_step(self)

			# Increment time
			t += stepper.dt
			self.time = t

			# Custom user function definition
			self.custom_user_function(self)

			# Print info
			self.print_info(physics, res, solver.itime, t, stepper.dt)

			# Write data file
			if (solver.itime + 1) % write_interval == 0:
				readwritedatafiles.write_data_file(self,
                        (solver.itime + 1) // write_interval)

			solver.itime += 1

		t1 = time.time()
		print("\nWall clock time = %g seconds" % (t1 - t0))
		print("--------------------------------------------------------" + \
				"-----------------------")
		self.wall_clock_time = t1 - t0

		if write_final_solution:
			readwritedatafiles.write_data_file(self, -1)
