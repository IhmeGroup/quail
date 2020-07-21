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
	def __init__(self, Params, physics, mesh):
		'''
		Method: __init__
		-------------------------------------------------------------------
		Initializes the DG object, verifies parameters, and initializes the state

		INPUTS:
			Params: list of parameters for the solver
			physics: solver object (current implementation supports Scalar and Euler equations)
			mesh: mesh object
		'''
		self.Params = Params
		self.physics = physics
		self.mesh = mesh
		self.DataSet = GenericData()

		self.Time = Params["StartTime"]
		self.nTimeStep = 0 # will be set later

		self.Stepper = stepper_tools.set_stepper(Params, physics.U)
		stepper_tools.set_time_stepping_approach(self.Stepper, Params)

		# Set the basis functions for the solver
		basis_name  = Params["InterpBasis"]
		self.basis = basis_tools.set_basis(physics.order, basis_name)

		node_type = Params["NodeType"]
		self.basis.get_1d_nodes = basis_tools.set_1D_node_calc(node_type)

		# Set quadrature
		self.basis.set_elem_quadrature_type(Params["ElementQuadrature"])
		self.basis.set_face_quadrature_type(Params["FaceQuadrature"])
		mesh.gbasis.set_elem_quadrature_type(Params["ElementQuadrature"])
		mesh.gbasis.set_face_quadrature_type(Params["FaceQuadrature"])

		self.basis.force_nodes_equal_quad_pts(Params["NodesEqualQuadpts"])

		# Limiter
		limiterType = Params["ApplyLimiter"]
		self.Limiter = limiter_tools.set_limiter(limiterType, physics.PHYSICS_TYPE)

		# Check validity of parameters
		self.check_compatibility()

		# Precompute operators
		self.precompute_matrix_operators()
		if self.Limiter is not None:
			self.Limiter.precompute_operators(self)
		physics.ConvFluxFcn.alloc_helpers(np.zeros([self.iface_operators.quad_wts.shape[0], physics.NUM_STATE_VARS]))

		# Initialize state
		if Params["RestartFile"] is None:
			self.init_state_from_fcn()

	def __repr__(self):
		return '{self.__class__.__name__}(Physics: {self.physics},\n   \
		Basis: {self.basis},\n   Stepper: {self.Stepper})'.format(self=self)

	def check_compatibility(self):
		mesh = self.mesh 
		Params = self.Params
		basis = self.basis

		# check for same shape between mesh and solution
		if mesh.gbasis.SHAPE_TYPE != basis.SHAPE_TYPE:
			raise errors.IncompatibleError

		if Params["InterpolateIC"] and basis.MODAL_OR_NODAL \
			!= ModalOrNodal.Nodal:
			raise errors.IncompatibleError

		# ----------------------------------------------------------------- #
		# Gauss Lobatto Nodes Compatibility Checks
		# ----------------------------------------------------------------- #
		node_type = Params["NodeType"]
		forcing_switch = Params["NodesEqualQuadpts"]
		elem_quad = Params["ElementQuadrature"]
		face_quad = Params["FaceQuadrature"]

		# Compatibility check for GLL nodes with triangles
		if NodeType[node_type] == NodeType.GaussLobatto and \
			basis.SHAPE_TYPE == ShapeType.Triangle:
			raise errors.IncompatibleError

		# Compatibility check for forcing nodes equal to quadrature points
		if NodeType[node_type] == NodeType.Equidistant and forcing_switch:
			raise errors.IncompatibleError
		if ( QuadratureType[elem_quad] != QuadratureType.GaussLobatto or \
			QuadratureType[face_quad] != QuadratureType.GaussLobatto ) \
			and forcing_switch:
			raise errors.IncompatibleError
		
		# ----------------------------------------------------------------- #


	@abstractmethod
	def precompute_matrix_operators(self):
		pass
	@abstractmethod
	def calculate_residual_elem(self, elem, Up, ER):
		pass

	@abstractmethod
	def calculate_residual_iface(self, iiface, UpL, UpR, RL, RR):
		pass

	@abstractmethod
	def calculate_residual_bface(self, ibfgrp, ibface, U, R):
		pass

	def init_state_from_fcn(self):
		mesh = self.mesh
		physics = self.physics
		basis = self.basis
		Params = self.Params
		iMM_elems = self.elem_operators.iMM_elems

		U = physics.U
		ns = physics.NUM_STATE_VARS
		order = physics.order

		if Params["InterpolateIC"]:
			eval_pts = basis.get_nodes(order)
		else:
			order = 2*np.amax([physics.order, 1])
			order = physics.QuadOrder(order)

			quad_order = basis.get_quadrature_order(mesh, order)
			quad_pts, quad_wts = basis.get_quadrature_data(quad_order)
			# quad_pts, quad_wts = segment.get_quadrature_gauss_legendre(quad_order)
			eval_pts = quad_pts
		npts = eval_pts.shape[0]

		for elem in range(mesh.nElem):
			xphys, _ = mesh_defs.ref_to_phys(mesh, elem, None, eval_pts)
			f = physics.CallFunction(physics.IC, x=xphys, t=self.Time)
			# f.shape = npts,ns
			if Params["InterpolateIC"]:
				solver_tools.interpolate_to_nodes(f, U[elem,:,:])
			else:
				solver_tools.L2_projection(mesh, iMM_elems[elem], basis, quad_pts, quad_wts, elem, f, U[elem,:,:])

	def project_state_to_new_basis(self, U_old, basis_old, order_old):
		mesh = self.mesh
		physics = self.physics
		basis = self.basis
		Params = self.Params
		iMM_elems = self.elem_operators.iMM_elems

		U = physics.U
		ns = physics.NUM_STATE_VARS

		# basis_old = basis_tools.set_basis(mesh, order_old, basis_name_old)
		if basis_old.SHAPE_TYPE != basis.SHAPE_TYPE:
			raise errors.IncompatibleError

		if Params["InterpolateIC"]:
			eval_pts = basis.get_nodes(physics.order)
		else:
			order = 2*np.amax([physics.order, order_old])
			quad_order = basis.get_quadrature_order(mesh, order)

			quad_pts, quad_wts = basis.get_quadrature_data(quad_order)
			eval_pts = quad_pts

		npts = eval_pts.shape[0]

		basis_old.get_basis_val_grads(eval_pts, get_val=True)

		for elem in range(mesh.nElem):
			# Up_old = np.matmul(basis_old.basis_val, U_old[elem,:,:])
			Up_old = helpers.evaluate_state(U_old[elem,:,:], basis_old.basis_val)

			if Params["InterpolateIC"]:
				solver_tools.interpolate_to_nodes(Up_old, U[elem,:,:])
			else:
				solver_tools.L2_projection(mesh, iMM_elems[elem], basis, quad_pts, quad_wts, elem, Up_old, U[elem,:,:])
	
	def calculate_residual(self, U, R):
		'''
		Method: calculate_residual
		-----------------------------------
		Calculates the boundary + volume integral for the DG formulation
		
		INPUTS:
			U: solution array
			
		OUTPUTS:
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

		self.calculate_residual_bfaces(U, R)
		self.calculate_residual_elems(U, R)
		self.calculate_residual_ifaces(U, R)

		return R

	def calculate_residual_elems(self, U, R):
		'''
		Method: calculate_residual_elems
		---------------------------------
		Calculates the volume integral across the entire domain
		
		INPUTS:
			U: solution array
			
		OUTPUTS:
			R: calculated residiual array
		'''
		mesh = self.mesh
		physics = self.physics

		for elem in range(mesh.nElem):
			R[elem] = self.calculate_residual_elem(elem, U[elem], R[elem])

	def calculate_residual_ifaces(self, U, R):
		'''
		Method: calculate_residual_ifaces
		-----------------------------------
		Calculates the boundary integral for all internal faces
		
		INPUTS:
			U: solution array
			
		OUTPUTS:
			R: calculated residual array (includes all face contributions)
		'''
		mesh = self.mesh
		physics = self.physics

		for iiface in range(mesh.nIFace):
			IFace = mesh.IFaces[iiface]
			elemL = IFace.ElemL
			elemR = IFace.ElemR
			faceL = IFace.faceL
			faceR = IFace.faceR

			UL = U[elemL]
			UR = U[elemR]
			RL = R[elemL]
			RR = R[elemR]

			RL, RR = self.calculate_residual_iface(iiface, UL, UR, RL, RR)

	def calculate_residual_bfaces(self, U, R):
		'''
		Method: calculate_residual_bfaces
		-----------------------------------
		Calculates the boundary integral for all the boundary faces
		
		INPUTS:
			U: solution array from internal cell
			
		OUTPUTS:
			R: calculated residual array from boundary face
		'''
		mesh = self.mesh
		physics = self.physics

		# for ibfgrp in range(mesh.nBFaceGroup):
		# 	BFG = mesh.BFaceGroups[ibfgrp]

		for BFG in mesh.BFaceGroups.values():

			for ibface in range(BFG.nBFace):
				BFace = BFG.BFaces[ibface]
				elem = BFace.Elem
				face = BFace.face

				R[elem] = self.calculate_residual_bface(BFG, ibface, U[elem], R[elem])


	def apply_time_scheme(self, fhistory=None):
		'''
		Method: apply_time_scheme
		-----------------------------
		Applies the specified time scheme to update the solution

		'''
		physics = self.physics
		mesh = self.mesh
		Order = self.Params["InterpOrder"]
		Stepper = self.Stepper
		Time = self.Time

		# Parameters
		TrackOutput = self.Params["TrackOutput"]
		WriteInterval = self.Params["WriteInterval"]
		if WriteInterval == -1:
			WriteInterval = np.NAN
		WriteFinalSolution = self.Params["WriteFinalSolution"]
		WriteInitialSolution = self.Params["WriteInitialSolution"]

		if WriteInitialSolution:
			ReadWriteDataFiles.write_data_file(self, 0)

		t0 = time.time()
		iwrite = 1

		# for iStep in range(Stepper.numtimesteps):
		iStep = 0
		while iStep < Stepper.numtimesteps:

			Stepper.dt = Stepper.get_time_step(Stepper, self)
			# Integrate in time
			# self.Time is used for local time
			R = Stepper.TakeTimeStep(self)

			# Increment time
			Time += Stepper.dt
			self.Time = Time

			# Info to print
			# PrintInfo = (iStep+1, self.Time, R.VectorNorm(ord=1))
			PrintInfo = (iStep+1, self.Time, np.linalg.norm(np.reshape(R,-1), ord=1))
			PrintString = "%d: Time = %g, Residual norm = %g" % (PrintInfo)

			# Output
			if TrackOutput:
				output,_ = post_defs.L2_error(mesh,physics,Time,"Entropy",False)
				OutputString = ", Output = %g" % (output)
				PrintString += OutputString

			# Print info
			print(PrintString)

			# Write to file if requested
			if fhistory is not None:
				fhistory.write("%d %g %g" % (PrintInfo))
				if TrackOutput:
					fhistory.write(" %g" % (output))
				fhistory.write("\n")

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
		Method: apply_limiter
		-------------------------
		Applies the limiter to the solution array, U.
		
		INPUTS:
			U: solution array

		OUTPUTS:
			U: limited solution array

		Notes: See Limiter.py for details
		'''
		if self.Limiter is not None:
			self.Limiter.limit_solution(self, U)


	def solve(self):
		'''
		Method: solve
		-----------------------------
		Performs the main solve of the DG method. Initializes the temporal loop. 

		'''
		mesh = self.mesh
		physics = self.physics
		basis = self.basis

		InterpOrder = self.Params["InterpOrder"]
		# numtimesteps = self.Params["NumTimeSteps"]
		# EndTime = self.Params["EndTime"]
		WriteTimeHistory = self.Params["WriteTimeHistory"]
		if WriteTimeHistory:
			fhistory = open("TimeHistory.txt", "w")
		else:
			fhistory = None


		''' Convert to lists '''
		# InterpOrder
		# if np.issubdtype(type(InterpOrder), np.integer):
		# 	InterpOrders = [InterpOrder]
		# elif type(InterpOrder) is list:
		# 	InterpOrders = InterpOrder 
		# else:
		# 	raise TypeError
		# nOrder = len(InterpOrders)
		# # nTimeStep
		# if np.issubdtype(type(nTimeStep), np.integer):
		# 	nTimeSteps = [nTimeStep]*nOrder
		# elif type(nTimeStep) is list:
		# 	nTimeSteps = nTimeStep 
		# else:
		# 	raise TypeError
		# # EndTime
		# if np.issubdtype(type(EndTime), np.floating):
		# 	EndTimes = []
		# 	for i in range(nOrder):
		# 		EndTimes.append(EndTime*(i+1))
		# elif type(EndTime) is list:
		# 	EndTimes = EndTime 
		# else:
		# 	raise TypeError


		# ''' Check compatibility '''
		# if nOrder != len(nTimeSteps) or nOrder != len(EndTimes):
		# 	raise ValueError

		# if np.any(np.diff(EndTimes) < 0.):
		# 	raise ValueError

		# if not OrderSequencing:
		# 	if len(InterpOrders) != 1:
		# 		raise ValueError

		''' Loop through Orders '''

		''' Compute time step '''
		# if nTimeStep != 0:
		# 	self.Stepper.dt = (EndTime-Time)/nTimeStep
		# self.numtimesteps = numtimesteps			

		''' Apply time scheme '''
		self.apply_time_scheme(fhistory)


		# for iOrder in range(nOrder):
		# 	Order = InterpOrders[iOrder]
		# 	''' Compute time step '''
		# 	if nTimeSteps[iOrder] != 0:
		# 		self.Stepper.dt = (EndTimes[iOrder]-Time)/nTimeSteps[iOrder]
		# 	self.nTimeStep = nTimeSteps[iOrder]

		# 	''' After first iteration, project solution to next Order '''
		# 	if iOrder > 0:
		# 		# Clear DataSet
		# 		delattr(self, "DataSet")
		# 		self.DataSet = GenericData()
		# 		# Increment Order
		# 		Order_old = physics.order
		# 		physics.order = Order
		# 		# Project
		# 		solver_tools.project_state_to_new_basis(self, mesh, physics, basis, Order_old)

		# 		basis.order = Order
		# 		basis.nb = basis.get_num_basis_coeff(Order)				

		# 		self.precompute_matrix_operators()

		# 	''' Apply time scheme '''
		# 	self.apply_time_scheme(fhistory)

		# 	Time = EndTimes[iOrder]


		if WriteTimeHistory:
			fhistory.close()		