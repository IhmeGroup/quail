# ------------------------------------------------------------------------ #
#
#       File : src/numerics/solver/base.py
#
#       Contains class definitions for the solver base class
#       available in the DG Python framework.
#
#       Authors: Eric Ching and Brett Bornhoft
#
#       Created: January 2020
#      
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import code
import copy
import numpy as np 
import time
import warnings

import errors
from data import ArrayList, GenericData

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

global echeck
echeck = -1


class SolverBase(ABC):
	'''
	This is a base class for any solver used in the DG Python framework

    Attributes:
    -----------
    Params: dictionary
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
    precompute_matrix_operators
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
	def __init__(self, Params, physics, mesh):
		self.Params = Params
		self.physics = physics
		self.mesh = mesh
		self.DataSet = GenericData()

		self.time = Params["StartTime"]
		self.num_time_steps = 0 # will be set later

		# Set the basis functions for the solver
		BASIS_TYPE  = Params["InterpBasis"]
		self.basis = basis_tools.set_basis(physics.order, BASIS_TYPE)

		NODE_TYPE = Params["NodeType"]
		self.basis.get_1d_nodes = basis_tools.set_1D_node_calc(NODE_TYPE)

		# Set quadrature
		self.basis.set_elem_quadrature_type(Params["ElementQuadrature"])
		self.basis.set_face_quadrature_type(Params["FaceQuadrature"])
		mesh.gbasis.set_elem_quadrature_type(Params["ElementQuadrature"])
		mesh.gbasis.set_face_quadrature_type(Params["FaceQuadrature"])

		self.basis.force_nodes_equal_quad_pts(Params["NodesEqualQuadpts"])

		# Limiter
		LIMITER_TYPE = Params["ApplyLimiter"]
		self.limiter = limiter_tools.set_limiter(LIMITER_TYPE, 
				physics.PHYSICS_TYPE)

	def __repr__(self):
		return '{self.__class__.__name__}(Physics: {self.physics},\n   \
		Basis: {self.basis},\n   Stepper: {self.Stepper})'.format(self=self)

	def check_compatibility(self):
		'''
		Checks a variety of parameter combinations to ensure compatibility
		for the specified input deck and the following simulation
		'''
		mesh = self.mesh 
		Params = self.Params
		basis = self.basis

		# check for same shape between mesh and solution
		if mesh.gbasis.SHAPE_TYPE != basis.SHAPE_TYPE:
			raise errors.IncompatibleError

		if not Params["L2InitialCondition"] and basis.MODAL_OR_NODAL \
			!= ModalOrNodal.Nodal:
			raise errors.IncompatibleError

		# Gauss Lobatto nodes compatibility checks
		node_type = Params["NodeType"]
		forcing_switch = Params["NodesEqualQuadpts"]
		elem_quad = Params["ElementQuadrature"]
		face_quad = Params["FaceQuadrature"]

		# compatibility check for GLL nodes with triangles
		if NodeType[node_type] == NodeType.GaussLobatto and \
			basis.SHAPE_TYPE == ShapeType.Triangle:
			raise errors.IncompatibleError

		# compatibility check for forcing nodes equal to quadrature points
		if NodeType[node_type] == NodeType.Equidistant and forcing_switch:
			raise errors.IncompatibleError
		if ( QuadratureType[elem_quad] != QuadratureType.GaussLobatto or \
			QuadratureType[face_quad] != QuadratureType.GaussLobatto ) \
			and forcing_switch:
			raise errors.IncompatibleError

	@abstractmethod
	def precompute_matrix_operators(self):
		'''
		Precomputes element and face helper functions that only need to be
		computed at the beginning of the simulation.
		'''
		pass
	@abstractmethod
	def get_element_residual(self, elem, Up, ER):
		'''
		Calculates the residual from the volume integral for each element
		
		Inputs:
		-------
			elem: element index
			Up: solution state

		Outputs:
		--------
			ER: calculated residiual array (for volume integral of specified 
			element)
		'''
		pass

	@abstractmethod
	def get_interior_face_residual(self, iiface, UpL, UpR, RL, RR):
		'''
		Calculates the surface integral for the internal faces
		
		Inputs:
		-------
			iiface: internal face index
			UpL: solution array from left neighboring element
			UpR: solution array from right neighboring element
			
		Outputs:
		--------
			RL: calculated residual array (left neighboring element 
			contribution)
			RR: calculated residual array (right neighboring element 
			contribution)
		'''
		pass

	@abstractmethod
	def get_boundary_face_residual(self, ibfgrp, ibface, U, R):
		'''
		Calculates the residual from the surface integral for each boundary 
		face

		Inputs:
		-------
			ibfgrp: index of BC group
			ibface: index of boundary face
			U: solution array from internal element
			
		Outputs:
		--------
			R: calculated residual array (from boundary face)
		'''
		pass

	def init_state_from_fcn(self):
		'''
		Initializes the state (initial condition) from the specified 
		function in the input deck. Either interpolates the state to the 
		nodes or uses an L2 projection to initialize the state.
		'''
		mesh = self.mesh
		physics = self.physics
		basis = self.basis
		Params = self.Params
		iMM_elems = self.elem_operators.iMM_elems

		U = physics.U
		ns = physics.NUM_STATE_VARS
		order = physics.order

		if not Params["L2InitialCondition"]:
			eval_pts = basis.get_nodes(order)
		else:
			order = 2*np.amax([physics.order, 1])
			order = physics.QuadOrder(order)

			quad_order = basis.get_quadrature_order(mesh, order)
			quad_pts, quad_wts = basis.get_quadrature_data(quad_order)

			eval_pts = quad_pts
		npts = eval_pts.shape[0]

		for elem in range(mesh.num_elems):
			xphys = mesh_tools.ref_to_phys(mesh, elem, eval_pts)
			f = physics.CallFunction(physics.IC, x=xphys, t=self.time)

			if not Params["L2InitialCondition"]:
				solver_tools.interpolate_to_nodes(f, U[elem,:,:])
			else:
				solver_tools.L2_projection(mesh, iMM_elems[elem], basis, quad_pts, quad_wts, elem, f, U[elem,:,:])

	def project_state_to_new_basis(self, U_old, basis_old, order_old):
		'''
		Projects the state of a restartfile onto a new basis/order of 
		accuracy

		Inputs:
		-------
			U_old: restart files old solution array
			basis_old: previous basis function
			order_old: previous polynomial order			
		'''
		mesh = self.mesh
		physics = self.physics
		basis = self.basis
		Params = self.Params
		iMM_elems = self.elem_operators.iMM_elems

		U = physics.U
		ns = physics.NUM_STATE_VARS

		if basis_old.SHAPE_TYPE != basis.SHAPE_TYPE:
			raise errors.IncompatibleError

		if not Params["L2InitialCondition"]:
			eval_pts = basis.get_nodes(physics.order)
		else:
			order = 2*np.amax([physics.order, order_old])
			quad_order = basis.get_quadrature_order(mesh, order)

			quad_pts, quad_wts = basis.get_quadrature_data(quad_order)
			eval_pts = quad_pts

		npts = eval_pts.shape[0]

		basis_old.get_basis_val_grads(eval_pts, get_val=True)

		for elem in range(mesh.num_elems):

			Up_old = helpers.evaluate_state(U_old[elem,:,:], 
					basis_old.basis_val)

			if not Params["L2InitialCondition"]:
				solver_tools.interpolate_to_nodes(Up_old, U[elem,:,:])
			else:
				solver_tools.L2_projection(mesh, iMM_elems[elem], basis, 
						quad_pts, quad_wts, elem, Up_old, U[elem,:,:])
	
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
		stepper = self.Stepper

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

		for elem in range(mesh.num_elems):
			R[elem] = self.get_element_residual(elem, U[elem], R[elem])

	def get_interior_face_residuals(self, U, R):
		'''
		Loops over the interior faces and calls the 
		get_interior_face_residual function for each face
		
		Inputs:
		-------
			U: solution array
			
		Outputs:
		--------
			R: calculated residual array (includes all face contributions)
		'''
		mesh = self.mesh
		physics = self.physics

		for iiface in range(mesh.num_interior_faces):
			IFace = mesh.interior_faces[iiface]
			elemL = IFace.elemL_id
			elemR = IFace.elemR_id
			faceL_id = IFace.faceL_id
			faceR_id = IFace.faceR_id

			UL = U[elemL]
			UR = U[elemR]
			RL = R[elemL]
			RR = R[elemR]

			RL, RR = self.get_interior_face_residual(iiface, UL, UR, RL, RR)

	def get_boundary_face_residuals(self, U, R):
		'''
		Loops over the boundary faces and calls the 
		get_boundary_face_residual function for each face
		
		Inputs:
		-------
			U: solution array
			
		Outputs:
		--------
			R: calculated residual array (includes all face contributions)
		'''
		mesh = self.mesh
		physics = self.physics

		for BFG in mesh.boundary_groups.values():

			for ibface in range(BFG.num_boundary_faces):
				boundary_face = BFG.boundary_faces[ibface]
				elem = boundary_face.elem_id
				face = boundary_face.face_id

				R[elem] = self.get_boundary_face_residual(BFG, ibface,
						U[elem], R[elem])


	def apply_time_scheme(self):
		'''
		Applies the specified time scheme to update the solution
		'''
		physics = self.physics
		mesh = self.mesh
		Order = self.Params["InterpOrder"]
		Stepper = self.Stepper
		Time = self.time

		# Parameters
		WriteInterval = self.Params["WriteInterval"]
		if WriteInterval == -1:
			WriteInterval = np.NAN
		WriteFinalSolution = self.Params["WriteFinalSolution"]
		WriteInitialSolution = self.Params["WriteInitialSolution"]

		if WriteInitialSolution:
			ReadWriteDataFiles.write_data_file(self, 0)

		t0 = time.time()
		iwrite = 1

		iStep = 0
		while iStep < Stepper.num_time_steps:

			Stepper.dt = Stepper.get_time_step(Stepper, self)
			# integrate in time
			R = Stepper.TakeTimeStep(self)

			# Increment time
			Time += Stepper.dt
			self.time = Time

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