# ------------------------------------------------------------------------ #
#
#       File : src/numerics/solver/base.py
#
#       Contains class definitions for the solver base class
#      
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import copy
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
	num_time_steps: int
		number of time steps required to solve to the specified end time
	basis: object
		contains all the information and methods for the basis class
	limiter: object
		contains all the information and methods for the limiter class

    Abstract Methods:
    -----------------
    precompute_matrix_helpers
    	precomputes a variety of functions and methods prior to running the
    	simulation
    get_element_residual
    	calculates the residual for a specific element
    get_interior_face_residual
    	calculates the residual for a specific face
    get_boundary_face_residual
    	calculates the residual for a specific boundary face
    
    Methods:
    --------
    check_compatibility
    	checks parameter compatibilities based on the given input deck
    init_state_from_fcn
    	initializes state from a specified function in the input deck
	project_state_to_new_basis
		takes a state from a restartfile and projects it onto a higher
		order of accuracy
	'''
	def __init__(self, params, physics, mesh):
		self.params = params
		self.physics = physics
		self.mesh = mesh

		self.time = params["InitialTime"]
		self.num_time_steps = 0 # will be set later

		# Set the basis functions for the solver
		BASIS_TYPE  = params["SolutionBasis"]
		self.basis = basis_tools.set_basis(physics.order, BASIS_TYPE)

		NODE_TYPE = params["NodeType"]
		self.basis.get_1d_nodes = basis_tools.set_1D_node_calc(NODE_TYPE)

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

		# check for same shape between mesh and solution
		if mesh.gbasis.SHAPE_TYPE != basis.SHAPE_TYPE:
			raise errors.IncompatibleError

		if not params["L2InitialCondition"] and basis.MODAL_OR_NODAL \
				!= ModalOrNodal.Nodal:
			raise errors.IncompatibleError

		# Gauss Lobatto nodes compatibility checks
		node_type = params["NodeType"]
		forcing_switch = params["ColocatedPoints"]
		elem_quad = params["ElementQuadrature"]
		face_quad = params["FaceQuadrature"]

		# compatibility check for GLL nodes with triangles
		if NodeType[node_type] == NodeType.GaussLobatto and \
			basis.SHAPE_TYPE == ShapeType.Triangle:
			raise errors.IncompatibleError

		# compatibility check for forcing nodes equal to quadrature points
		if NodeType[node_type] == NodeType.Equidistant and forcing_switch:
			raise errors.IncompatibleError
		if (QuadratureType[elem_quad] != QuadratureType.GaussLobatto or \
				QuadratureType[face_quad] != QuadratureType.GaussLobatto) \
				and forcing_switch:
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

		U = physics.U
		ns = physics.NUM_STATE_VARS
		order = physics.order

		# Get solution nodes or quadrature info
		if not params["L2InitialCondition"]:
			# Solution nodes
			eval_pts = basis.get_nodes(order)
		else:
			# Quadrature
			order = 2*np.amax([physics.order, 1])
			order = physics.get_quadrature_order(order)

			quad_order = basis.get_quadrature_order(mesh, order)
			quad_pts, quad_wts = basis.get_quadrature_data(quad_order)

			eval_pts = quad_pts
		npts = eval_pts.shape[0]

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
						quad_pts, quad_wts, elem_ID, f, U[elem_ID,:,:])

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

		U = physics.U
		ns = physics.NUM_STATE_VARS

		if basis_old.SHAPE_TYPE != basis.SHAPE_TYPE:
			raise errors.IncompatibleError

		if not params["L2InitialCondition"]:
			eval_pts = basis.get_nodes(physics.order)
		else:
			order = 2*np.amax([physics.order, order_old])
			quad_order = basis.get_quadrature_order(mesh, order)

			quad_pts, quad_wts = basis.get_quadrature_data(quad_order)
			eval_pts = quad_pts

		npts = eval_pts.shape[0]

		basis_old.get_basis_val_grads(eval_pts, get_val=True)

		for elem_ID in range(mesh.num_elems):

			Up_old = helpers.evaluate_state(U_old[elem_ID,:,:], 
					basis_old.basis_val)

			if not params["L2InitialCondition"]:
				solver_tools.interpolate_to_nodes(Up_old, U[elem_ID,:,:])
			else:
				solver_tools.L2_projection(mesh, iMM_elems[elem_ID], basis, 
						quad_pts, quad_wts, elem_ID, Up_old, U[elem_ID,:,:])
	
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

		if R is None:
			R = np.copy(U)
		# Initialize residual to zero

		if stepper.balance_const is None:
			R[:] = 0.
		else:
			R[:] = stepper.balance_const

		self.get_boundary_face_residuals(U, R)
		self.get_element_residuals(U, R)
		self.get_interior_face_residuals(U, R)

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
		mesh = self.mesh
		physics = self.physics

		for elem_ID in range(mesh.num_elems):
			R[elem_ID] = self.get_element_residual(elem_ID, U[elem_ID], R[elem_ID])

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
		physics = self.physics

		for int_face_ID in range(mesh.num_interior_faces):
			interior_face = mesh.interior_faces[int_face_ID]
			elemL_ID = interior_face.elemL_ID
			elemR_ID = interior_face.elemR_ID

			Uc_L = U[elemL_ID] # state coeffs of "left" element
			Uc_R = U[elemR_ID] # state coeffs of "right" element
			R_L = R[elemL_ID]
			R_R = R[elemR_ID]

			R_L, R_R = self.get_interior_face_residual(int_face_ID, Uc_L, 
					Uc_R, R_L, R_R)

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

		for BFG in mesh.boundary_groups.values():

			for bface_ID in range(BFG.num_boundary_faces):
				boundary_face = BFG.boundary_faces[bface_ID]
				elem_ID = boundary_face.elem_ID
				face = boundary_face.face_ID

				R[elem_ID] = self.get_boundary_face_residual(BFG, bface_ID,
						U[elem_ID], R[elem_ID])


	def apply_time_scheme(self):
		'''
		Applies the specified time scheme to update the solution
		'''
		physics = self.physics
		mesh = self.mesh
		Order = self.params["SolutionOrder"]
		stepper = self.stepper
		t = self.time

		# Parameters
		WriteInterval = self.params["WriteInterval"]
		if WriteInterval == -1:
			WriteInterval = np.NAN
		WriteFinalSolution = self.params["WriteFinalSolution"]
		WriteInitialSolution = self.params["WriteInitialSolution"]

		if WriteInitialSolution:
			ReadWriteDataFiles.write_data_file(self, 0)

		t0 = time.time()
		iwrite = 1

		iStep = 0
		while iStep < stepper.num_time_steps:

			stepper.dt = stepper.get_time_step(stepper, self)
			# integrate in time
			R = stepper.take_time_step(self)

			# Increment time
			t += stepper.dt
			self.time = t

			# Info to print
			PrintInfo = (iStep+1, self.time, \
					np.linalg.norm(np.reshape(R,-1), ord=1))
			PrintString = "%d: Time = %g, Residual norm = %g" % (PrintInfo)

			# Print info
			print(PrintString)

			# Write data file
			if (iStep + 1) % WriteInterval == 0:
				ReadWriteDataFiles.write_data_file(self, iwrite)
				iwrite += 1

			iStep += 1

		t1 = time.time()
		print("Wall clock time = %g seconds" % (t1-t0))

		if WriteFinalSolution:
			ReadWriteDataFiles.write_data_file(self, -1)
			

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


	def solve(self):
		'''
		Performs the main solve of the DG method. Initializes the temporal
		loop. 
		'''	

		# apply time scheme
		self.apply_time_scheme()	