from abc import ABC, abstractmethod
import code
import numpy as np 
from scipy.optimize import fsolve, root

from solver.tools import mult_inv_mass_matrix
import solver.tools as solver_tools

class ODESolvers():

	class ODEBase(ABC):
		def __init__(self, U):
			# self.TimeStep = dt
			self.R = np.zeros_like(U)
			self.dt = 0.
			self.numtimesteps = 0
			self.get_time_step = None
			self.balance_const = None
		def __repr__(self):
			return '{self.__class__.__name__}(TimeStep={self.dt})'.format(self=self)

	class BDF1(ODEBase):

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
					A[elem],iA[elem] = self.get_jacobian_matrix_elem(solver, elem, iMM_elems[elem], Up[elem])
				return A, iA

		def get_jacobian_matrix_elem(self, solver, elem, iMM, Up):

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
			'''
			Evaluate the source term jacobian
			'''
			jac = np.zeros([nq,ns,ns])
			# for Source in Sources:
			# 	jac += Source.get_jacobian(Uq)
			jac = physics.SourceJacobianState(nq, x, solver.Time, Uq, jac) # [nq,ns]

			dRdU = solver_tools.calculate_dRdU(elem_ops, elem, jac)

			A = np.expand_dims(np.eye(nb), axis=2) - beta*dt*np.einsum('ij,jkl->ijl',iMM,dRdU)
			iA = np.zeros_like(A)
			for s in range(ns):
				iA[:,:,s] = np.linalg.inv(A[:,:,s])
			return A, iA

	class Trapezoidal(BDF1):
		BETA = 0.5

		def TakeTimeStep(self, solver):
			super().TakeTimeStep(solver)