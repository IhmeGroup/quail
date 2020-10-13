# ------------------------------------------------------------------------ #
#
#       File : src/numerics/timestepping/ode.py
#
#       Contains ODE solvers for operator splitting schemes in stepper.py
#      
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np 
from scipy.optimize import fsolve, root

from solver.tools import mult_inv_mass_matrix
import solver.tools as solver_tools


class ODESolvers():
	'''
	ODESolvers is a class of classes used as a Mixin class for operator 
	splitting approaches (see stepper.py). These methods are specifically
	used for solving ODEs of the form:

		dU/dt = S(U)

	Current schemes supported include:
		- Backward Difference (BDF1)
		- Trapezoidal Scheme (Trapezoidal)
	'''
	class ODEBase(ABC):
		'''
		This is an abstract base class used to represent a specific ODE 
		solver for operator splitting integration schemes.

		Attributes:
		-----------
		R: numpy array of floats (shape : [nelem, nb, ns])
			solution's residaul array
		dt: float
			time-step for the solution
		num_time_steps: int
			number of time steps for the given solution's FinalTime
		get_time_step: method
			method to obtain dt given input decks logic (CFL-based vs # of 
			timesteps, etc...)
		balance_const: numpy array of floats (shaped like R)
			balancing constant array used only with the Simpler splitting 
			scheme
		
		Abstract Methods:
		-----------------
		take_time_step
			method that takes a given time step for the solver depending on 
			the selected time-stepping scheme
		'''
		def __init__(self, U):
			self.R = np.zeros_like(U)
			self.dt = 0.
			self.num_time_steps = 0
			self.get_time_step = None
			self.balance_const = None

		def __repr__(self):
			return '{self.__class__.__name__}(TimeStep={self.dt})'.format( \
					self=self)

	class BDF1(ODEBase):
		'''
		1st-order Backward Differencing (BDF1) method inherits attributes 
		from ODEBase. See ODEBase for detailed comments of methods and 
		attributes.

		Additional methods and attributes are commented below.
		''' 

		# constant used to differentiate between BDF1 and Trapezoidal scheme
		BETA = 1.0

		def take_time_step(self, solver):
			mesh = solver.mesh
			U = solver.state_coeffs

			R = self.R

			R = solver.get_residual(U, R)
			dU = mult_inv_mass_matrix(mesh, solver, self.dt, R)

			A, iA = self.get_jacobian_matrix(mesh, solver)

			R = np.einsum('ijkl,ikl->ijl',A,U) + dU
			U = np.einsum('ijkl,ikl->ijl',iA,R)

			solver.apply_limiter(U)
			solver.state_coeffs = U

			return R

		def get_jacobian_matrix(self, mesh, solver):
			'''
			Calculates the Jacobian matrix of the source term and its inverse for all elements

			Inputs:
			-------
				mesh: mesh object
				solver: solver object (e.g., DG, ADERDG, etc...)

			Outputs:
			-------- 
				A: matrix returned for linear solve [nelem, nb, nb, ns]
				iA: inverse matrix returned for linear solve 
					[nelem, nb, nb, ns]
			'''
			basis = solver.basis
			nb = basis.nb
			physics = solver.physics
			U = solver.state_coeffs
			ns = physics.NUM_STATE_VARS

			iMM_elems = solver.elem_helpers.iMM_elems
			
			A = np.zeros([mesh.num_elems, nb, nb, ns])
			iA = np.zeros([mesh.num_elems, nb, nb, ns])

			for elem_ID in range(mesh.num_elems):
				A[elem_ID], iA[elem_ID] = self.get_jacobian_matrix_elem(solver, 
						elem_ID, iMM_elems[elem_ID], U[elem_ID])

			return A, iA # [nelem, nb, nb, ns]

		def get_jacobian_matrix_elem(self, solver, elem_ID, iMM, Uc):
			'''
			Calculates the Jacobian matrix of the source term and its 
			inverse for each element. Definition of 'Jacobian' matrix:

			A = I - BETA*dt*iMM^{-1}*dRdU

			Inputs:
			-------
				solver: solver object (e.g., DG, ADERDG, etc...)
				elem_ID: element ID
				iMM: inverse mass matrix [nb, nb]
				Uc: state coefficients on element [nb, ns]

			Outputs:
			-------- 
				A: matrix returned for linear solve [nelem, nb, nb, ns]
				iA: inverse matrix returned for linear solve 
					[nelem, nb, nb, ns]
			'''
			beta = self.BETA
			dt = solver.Stepper.dt
			physics = solver.physics
			source_terms = physics.source_terms

			elem_helpers = solver.elem_helpers
			basis_val = elem_helpers.basis_val
			quad_wts = elem_helpers.quad_wts
			x_elems = elem_helpers.x_elems
			x = x_elems[elem_ID]
			nq = quad_wts.shape[0]
			ns = physics.NUM_STATE_VARS
			nb = basis_val.shape[1]
			Uq = np.matmul(basis_val, Uc)

			# evaluate the source term Jacobian [nq, ns, ns]
			Sjac = np.zeros([nq,ns,ns])
			Sjac = physics.eval_source_term_jacobians(Uq, x, solver.time, Sjac) 

			# call solver helper to get dRdU (see solver/tools.py)
			dRdU = solver_tools.calculate_dRdU(elem_helpers, elem_ID, Sjac)

			A = np.expand_dims(np.eye(nb), axis=2) - beta*dt * \
					np.einsum('ij,jkl->ijl',iMM,dRdU)
			iA = np.zeros_like(A)

			for s in range(ns):
				iA[:,:,s] = np.linalg.inv(A[:,:,s])

			return A, iA # [nb, nb, ns]

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