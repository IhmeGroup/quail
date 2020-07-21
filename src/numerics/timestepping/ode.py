# ------------------------------------------------------------------------ #
#
#       File : numerics/timestepping/ode.py
#
#       Contains ODE solvers for operator splitting schemes in stepper.py
#
#       Authors: Eric Ching and Brett Bornhoft
#
#       Created: January 2020
#      
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import code
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

		Attributes
		-----------
		R : numpy array of floats (shape : [nelem, nb, ns])
			solution's residaul array
		dt : float
			time-step for the solution
		numtimesteps : int
			number of time steps for the given solution's endtime
		get_time_step : method
			method to obtain dt given input decks logic (CFL-based vs # of 
			timesteps, etc...)
		balance_const : numpy array of floats (shaped like R)
			balancing constant array used only with the Simpler splitting 
			scheme
		
		Abstract Methods:
		------------------
		TakeTimeStep
			method that takes a given time step for the solver depending on 
			the selected time-stepping scheme
		'''
		def __init__(self, U):
			'''
			Attributes
			-----------
			R : numpy array of floats (shape : [nelem, nb, ns])
				solution's residaul array
			dt : float
				time-step for the solution
			numtimesteps : int
				number of time steps for the given solution's endtime
			get_time_step : method
				method to obtain dt given input decks logic (CFL-based vs # 
				of timesteps, etc...)
			balance_const : numpy array of floats (shaped like R)
				balancing constant array used only with the Simpler
				splitting scheme
			'''
			self.R = np.zeros_like(U)
			self.dt = 0.
			self.numtimesteps = 0
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

		def TakeTimeStep(self, solver):

			physics = solver.physics
			DataSet = solver.DataSet
			mesh = solver.mesh
			U = physics.U

			R = self.R

			R = solver.calculate_residual(U, R)
			dU = mult_inv_mass_matrix(mesh, solver, self.dt, R)

			A, iA = self.get_jacobian_matrix(mesh, solver)

			R = np.einsum('ijkl,ikl->ijl',A,U) + dU
			U = np.einsum('ijkl,ikl->ijl',iA,R)

			solver.apply_limiter(U)
			physics.U = U

			return R

		def get_jacobian_matrix(self, mesh, solver):
			'''
			Method: get_jacobian_matrix
			----------------------------
			Calculates the jacobian matrix of the source term and its inverse for all elements

			INPUTS:
				mesh: mesh object
				solver: solver object (i.e. DG, ADERDG, etc...)

			OUTPUTS: 
				A: matrix returned for linear solve [nelem, nb, nb, ns]
				iA: inverse matrix returned for linear solve 
					[nelem, nb, nb, ns]
			'''
			basis = solver.basis
			nb = basis.nb
			DataSet = solver.DataSet
			physics = solver.physics
			Up = physics.U
			ns = physics.NUM_STATE_VARS

			iMM_elems = solver.elem_operators.iMM_elems
			
			A = np.zeros([mesh.nElem, nb, nb, ns])
			iA = np.zeros([mesh.nElem, nb, nb, ns])

			for elem in range(mesh.nElem):
				A[elem], iA[elem] = self.get_jacobian_matrix_elem(solver, 
					elem, iMM_elems[elem], Up[elem])

			return A, iA # [nelem, nb, nb, ns]

		def get_jacobian_matrix_elem(self, solver, elem, iMM, Up):
			'''
			Method: get_jacobian_matrix_elem
			--------------------------------
			Calculates the jacobian matrix of the source term and its 
			inverse for each element. Definition of 'jacobian' matrix:

			A = I - BETA*dt*iMM^{-1}*dRdU

			INPUTS:
				solver: solver object (i.e. DG, ADERDG, etc...)
				elem: element index [int]
			OUTPUTS: 
				A: matrix returned for linear solve [nelem, nb, nb, ns]
				iA: inverse matrix returned for linear solve 
					[nelem, nb, nb, ns]
			'''
			beta = self.BETA
			dt = solver.Stepper.dt
			physics = solver.physics
			Sources = physics.Sources

			elem_ops = solver.elem_operators
			basis_val = elem_ops.basis_val
			quad_wts = elem_ops.quad_wts
			x_elems = elem_ops.x_elems
			x = x_elems[elem]
			nq = quad_wts.shape[0]
			ns = physics.NUM_STATE_VARS
			nb = basis_val.shape[1]
			Uq = np.matmul(basis_val, Up)

			# evaluate the source term jacobian [nq, ns, ns]
			jac = np.zeros([nq,ns,ns])
			jac = physics.SourceJacobianState(nq, x, solver.Time, Uq, jac) 

			# call solver helper to get dRdU (see solver/tools.py)
			dRdU = solver_tools.calculate_dRdU(elem_ops, elem, jac)

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

		def TakeTimeStep(self, solver):
			super().TakeTimeStep(solver)