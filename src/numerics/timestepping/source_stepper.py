 # ------------------------------------------------------------------------ #
#
#       File : src/numerics/timestepping/source_stepper.py
#
#       Contains solvers for operator splitting schemes in stepper.py
#      
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np

import numerics.helpers.helpers as helpers

from solver.tools import mult_inv_mass_matrix
import solver.tools as solver_tools


class SourceSolvers():
	'''
	SourceSolvers is a class of classes used as a Mixin class for operator 
	splitting approaches (see stepper.py). These methods are specifically
	used for solving ODEs of the form:
		dU/dt = S(U)
	Current schemes supported include:
		- Backward Difference (BDF1)
		- Trapezoidal Scheme (Trapezoidal)
	'''
	class SourceStepperBase(ABC):
		'''
		This is an abstract base class used to represent a specific ODE 
		solver for operator splitting integration schemes.
		Attributes:
		-----------
		res: numpy array of floats (shape : [nelem, nb, ns])
			solution's residaul array
		dt: float
			time-step for the solution
		num_time_steps: int
			number of time steps for the given solution's FinalTime
		get_time_step: method
			method to obtain dt given input decks logic (CFL-based vs # of 
			timesteps, etc...)
		balance_const: numpy array of floats (shaped like res)
			balancing constant array used only with the Simpler splitting 
			scheme
		
		Abstract Methods:
		-----------------
		take_time_step
			method that takes a given time step for the solver depending on 
			the selected time-stepping scheme
		'''
		def __init__(self, U):
			self.res = np.zeros_like(U)
			self.dt = 0.
			self.num_time_steps = 0
			self.get_time_step = None
			self.balance_const = None

		def __repr__(self):
			return '{self.__class__.__name__}(TimeStep={self.dt})'.format( \
					self=self)

	class BDF1(SourceStepperBase):
		'''
		1st-order Backward Differencing (BDF1) method inherits attributes 
		from SourceStepperBase. See SourceStepperBase for detailed comments of methods and 
		attributes.

		Additional methods and attributes are commented below.
		''' 

		# constant used to differentiate between BDF1 and Trapezoidal scheme
		BETA = 1.0

		def take_time_step(self, solver):
			mesh = solver.mesh
			U = solver.state_coeffs

			res = self.res

			res = solver.get_residual(U, res)
			dU = mult_inv_mass_matrix(mesh, solver, self.dt, res)

			A, iA = self.get_jacobian_matrix(mesh, solver)

			res = np.einsum('ijkll, ijl -> ikl', A, U) + dU
			U = np.einsum('ijkll, ijl -> ikl', iA, res)

			solver.apply_limiter(U)
			solver.state_coeffs = U

			return res # [ne, nb, ns]

		def get_jacobian_matrix(self, mesh, solver):
			'''
			Calculates the Jacobian matrix of the source term and its inverse for all elements
			Inputs:
			-------
				mesh: mesh object
				solver: solver object (e.g., DG, ADERDG, etc...)
			Outputs:
			-------- 
				A: matrix returned for linear solve [ne, nb, nb, ns, ns]
				iA: inverse matrix returned for linear solve 
					[ne, nb, nb, ns, ns]
			'''
			basis = solver.basis
			nb = basis.nb
			physics = solver.physics
			U = solver.state_coeffs
			ns = physics.NUM_STATE_VARS

			iMM_elems = solver.elem_helpers.iMM_elems

			A, iA = self.get_jacobian_matrix_elems(solver, iMM_elems, U)

			return A, iA # [nelem, nb, nb, ns]

		def get_jacobian_matrix_elems(self, solver, iMM_elems, Uc):
			'''
			Calculates the Jacobian matrix of the source term and its 
			inverse for each element. Definition of 'Jacobian' matrix:
			A = I - BETA*dt*iMM^{-1}*dRdU

			Inputs:
			-------
				solver: solver object (e.g., DG, ADERDG, etc...)
				elem_ID: element ID
				iMM: inverse mass matrix [ne, nb, nb]
				Uc: state coefficients [ne, nb, ns]
			Outputs:
			-------- 
				A: matrix returned for linear solve [ne, nb, nb, ns, ns]
				iA: inverse matrix returned for linear solve 
					[ne, nb, nb, ns, ns]
			'''
			mesh = solver.mesh
			nelem = mesh.num_elems
			beta = self.BETA
			dt = solver.stepper.dt
			physics = solver.physics
			source_terms = physics.source_terms

			elem_helpers = solver.elem_helpers
			basis_val = elem_helpers.basis_val
			quad_wts = elem_helpers.quad_wts
			x_elems = elem_helpers.x_elems
			djac_elems = elem_helpers.djac_elems
			nq = quad_wts.shape[0]
			ns = physics.NUM_STATE_VARS
			nb = basis_val.shape[1]

			Uq = helpers.evaluate_state(Uc, basis_val,
					skip_interp=solver.basis.skip_interp) # [ne, nq, ns])
			
			# Evaluate the source term Jacobian [ne, nq, ns, ns]
			Sjac = np.zeros([nelem, nq, ns, ns])
			Sjac = physics.eval_source_term_jacobians(Uq, x_elems, 
					solver.time, Sjac) 

			# Call solver helper to get dRdU (see solver/tools.py)
			dRdU = solver_tools.calculate_dRdU(elem_helpers, Sjac)
				# [ne, nb, nb, ns, ns]

			# Define the identity matrix
			I = np.expand_dims(np.expand_dims(np.eye(nb), axis=2), axis=3)

			A = I - beta*dt * \
					np.einsum('eij, ejklm -> eiklm', iMM_elems, dRdU)

			iA = np.zeros_like(A)
			for i in range(ns):
				for s in range(ns):
					iA[:, :, :, i, s] = np.linalg.inv(A[:, :, :, i, s])
			
			return A, iA # [ne, nb, nb, ns, ns]


	class Trapezoidal(BDF1):
		'''
		2nd-order Trapezoidal method inherits attributes 
		from BDF1. See BDF1 for detailed comments of methods and 
		attributes.
		Note: Trapezoidal method is identical to linearized BDF1 other than 
		the 'BETA' constant.
		''' 
		BETA = 0.5 

		def take_time_step(self, solver):
			super().take_time_step(solver)