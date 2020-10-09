# ------------------------------------------------------------------------ #
#
#       File : src/numerics/solver/base.py
#
#       Contains class definitions for the solver base class
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np
import time

import errors

from general import ModalOrNodal, NodeType, ShapeType, QuadratureType

import meshing.meshbase as mesh_defs
import meshing.tools as mesh_tools

import numerics.basis.tools as basis_tools

import numerics.helpers.helpers as helpers

import numerics.limiting.tools as limiter_tools

import numerics.timestepping.tools as stepper_tools
import numerics.timestepping.stepper as stepper_defs
import numerics.quadrature.segment as segment

import processing.post as post_defs
import processing.readwritedatafiles as ReadWriteDataFiles

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
    	calculates the residual contribution for  a given element interior
    get_interior_face_residual
    	calculates the residual contribution for a specific interior face
    get_boundary_face_residual
    	calculates the residual contribution for a specific boundary face

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
		limiter_type = params["ApplyLimiter"]
		self.limiter = limiter_tools.set_limiter(limiter_type,
				physics.PHYSICS_TYPE)

		# Console output
		self.verbose = params["Verbose"]
		self.min_state = np.zeros(physics.NUM_STATE_VARS)
		self.max_state = np.zeros(physics.NUM_STATE_VARS)

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

	@abstractmethod
	def precompute_matrix_helpers(self):
		'''
		Precomputes element and face helper arrays and structures that only
		need to be computed once at the beginning of the simulation.
		'''
		pass

	@abstractmethod
	def get_element_residual(self, elem_ID, Uc, R_elem):
		'''
		Calculates the volume contribution to the residual for a given
		element.

		Inputs:
		-------
			elem_ID: element index
			Up: solution state
			R_elem: residual array

		Outputs:
		--------
			R_elem: calculated residual array
		'''
		pass

	@abstractmethod
	def get_interior_face_residual(self, int_face_ID, Uc_L, Uc_R, R_L, R_R):
		'''
		Calculates the surface integral for the interior faces

		Inputs:
		-------
			int_face_ID: interior face ID
			Uc_L: solution array for left neighboring element (polynomial
				coefficients)
			Uc_R: solution array for right neighboring element (polynomial
				coefficients)
			R_L: residual array (left neighboring element)
			R_R: residual array (right neighboring element)

		Outputs:
		--------
			R_L: calculated residual array (left neighboring element
			contribution)
			R_R: calculated residual array (right neighboring element
			contribution)
		'''
		pass

	@abstractmethod
	def get_boundary_face_residual(self, bgroup, bface_ID, Uc, R_B):
		'''
		Calculates the residual from the surface integral for each boundary
		face

		Inputs:
		-------
			bgroup: boundary group object
			bface_ID: ID of boundary face
			Uc: solution array from adjacent element
			R_B: residual array (for adjacent element)

		Outputs:
		--------
			R_B: calculated residual array (from boundary face)
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

		for elem_ID in range(mesh.num_elems):
			# Compute state
			xphys = mesh_tools.ref_to_phys(mesh, elem_ID, eval_pts)
			f = physics.IC.get_state(physics, x=xphys, t=self.time)

			if not params["L2InitialCondition"]:
				# Interpolate to solution nodes
				solver_tools.interpolate_to_nodes(f, U[elem_ID,:,:])
			else:
				# L2 projection
				solver_tools.L2_projection(mesh, iMM_elems[elem_ID], basis,
						quad_pts, quad_wts, elem_ID, f, U[elem_ID, :, :])

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

		for elem_ID in range(mesh.num_elems):
			Up_old = helpers.evaluate_state(U_old[elem,:,:],
					basis_old.basis_val)

			if not params["L2InitialCondition"]:
				solver_tools.interpolate_to_nodes(Uq_old, U[elem_ID,:,:])
			else:
				solver_tools.L2_projection(mesh, iMM_elems[elem_ID], basis,
						quad_pts, quad_wts, elem_ID, Uq_old,
						U[elem_ID, :, :])

	def get_residual(self, U, R):
		'''
		Calculates the surface + volume integral for the DG formulation

		Inputs:
		-------
			U: solution array

		Outputs:
		--------
			R: residual array
		'''
		mesh = self.mesh
		physics = self.physics
		stepper = self.stepper

		# Initialize residual to zero
		if stepper.balance_const is None:
			R[:] = 0.
		else:
			R[:] = stepper.balance_const

		#self.get_boundary_face_residuals(U, R)
		self.get_element_residuals(U, R)
		#self.get_interior_face_residuals(U, R)

		return R

	def get_element_residuals(self, U, R):
		'''
		Loops over the elements and calls the get_element_residual
		function for each element

		Inputs:
		-------
			U: solution array

		Outputs:
		--------
			R: calculated residiual array
		'''

		R = self.get_element_residual(U, R)

	def get_interior_face_residuals(self, U, R):
		'''
		Loops over the interior faces and calls the
		get_interior_face_residual function for each face

		Inputs:
		-------
			U: solution array
			R: residual array

		Outputs:
		--------
			R: calculated residual array (includes all interior face
				contributions)
		'''
		mesh = self.mesh

		# TODO: refactor this loop out
		UL = np.empty((mesh.num_interior_faces,) + U.shape[1:]) # [nf, nb, ns]
		UR = np.empty((mesh.num_interior_faces,) + U.shape[1:]) # [nf, nb, ns]
		RL = np.empty((mesh.num_interior_faces,) + U.shape[1:]) # [nf, nb, ns]
		RR = np.empty((mesh.num_interior_faces,) + U.shape[1:]) # [nf, nb, ns]
		elemL = np.empty(mesh.num_interior_faces, dtype=int)
		elemR = np.empty(mesh.num_interior_faces, dtype=int)
		faceL_id = np.empty(mesh.num_interior_faces, dtype=int)
		faceR_id = np.empty(mesh.num_interior_faces, dtype=int)
		for iiface in range(mesh.num_interior_faces):
			IFace = mesh.interior_faces[iiface]
			elemL[iiface] = IFace.elemL_id
			elemR[iiface] = IFace.elemR_id
			faceL_id[iiface] = IFace.faceL_id
			faceR_id[iiface] = IFace.faceR_id
		UL = U[elemL]
		UR = U[elemR]
		RL = R[elemL]
		RR = R[elemR]

		# TODO: This RL and RR probably have to be manually added to R, because
		# (I think) it's actually a copy of R, not a view of R.
		RL, RR = self.get_interior_face_residual(faceL_id, faceR_id, UL, UR, RL, RR)
		R[elemL] = RL
		R[elemR] = RR

	def get_boundary_face_residuals(self, U, R):
		'''
		Loops over the boundary faces and calls the
		get_boundary_face_residual function for each face

		Inputs:
		-------
			U: solution array
			R: residual array

		Outputs:
		--------
			R: calculated residual array (includes all boundary face
				contributions)
		'''
		mesh = self.mesh
		physics = self.physics

		# Loop through boundary groups
		for bgroup in mesh.boundary_groups.values():
			# Loop through boundary faces
			for bface_ID in range(bgroup.num_boundary_faces):
				boundary_face = bgroup.boundary_faces[bface_ID]
				elem_ID = boundary_face.elem_ID

				R[elem_ID] = self.get_boundary_face_residual(bgroup,
						bface_ID, U[elem_ID], R[elem_ID])

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
		if self.limiter is not None:
			self.limiter.limit_solution(self, U)

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
		self.min_state = np.minimum(self.min_state, np.amin(Uq, axis=0))
		self.max_state = np.maximum(self.max_state, np.amax(Uq, axis=0))

	def print_info(self, physics, R, itime, t, dt):
		'''
		Prints key information to console. If self.verbose is False, then
		only time and residual info is printed; otherwise, min and max
		values of the state are also reported.

		Inputs:
		-------
			physics: physics object
			R: residual array [num_elems, nb, ns]
			itime: time iteration
			t: time
			dt: time step size
		'''
		# Basic info: time, residual
		print("%d: Time = %g - Time step = %g - Residual norm = %g" % (
				itime + 1, t, dt, np.linalg.norm(np.reshape(R, -1), ord=1)))

		# If requested, report min and max of state variables
		if self.verbose:
			print("\nMin|Max at volume quadrature points:")
			s = 0
			for state_var in physics.StateVariables:
				string = "    " + state_var.name + ": " + "%g | %g"
				print(string % (self.min_state[s], self.max_state[s]))
				s += 1

		print("--------------------------------------------------------" + \
				"-----------------------")

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
			ReadWriteDataFiles.write_data_file(self, 0)

		t0 = time.time()
		iwrite = 1

		print("\n\nUNSTEADY SOLVE:")
		print("--------------------------------------------------------" + \
				"-----------------------")

		itime = 0
		while itime < stepper.num_time_steps:
			# Reset min and max state
			self.max_state[:] = -np.inf
			self.min_state[:] = np.inf

			# Get time step size
			stepper.dt = stepper.get_time_step(stepper, self)

			# Integrate in time
			R = stepper.take_time_step(self)

			# Increment time
			t += stepper.dt
			self.time = t

			# Print info
			self.print_info(physics, R, itime, t, stepper.dt)

			# Write data file
			if (itime + 1) % write_interval == 0:
				ReadWriteDataFiles.write_data_file(self, iwrite)
				iwrite += 1

			itime += 1

		t1 = time.time()
		print("\nWall clock time = %g seconds" % (t1 - t0))
		print("--------------------------------------------------------" + \
				"-----------------------")

		if write_final_solution:
			ReadWriteDataFiles.write_data_file(self, -1)
