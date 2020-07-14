from abc import ABC, abstractmethod
import code
import numpy as np 

from data import ArrayList
from general import StepperType, ODESolverType
from solver.tools import mult_inv_mass_matrix
import numerics.basis.tools as basis_tools
import solver.tools as solver_tools

class StepperBase(ABC):
	def __init__(self, U):
		# self.TimeStep = dt
		self.R = np.zeros_like(U)
		self.dt = 0.
	def __repr__(self):
		return '{self.__class__.__name__}(TimeStep={self.dt})'.format(self=self)
	@abstractmethod
	def TakeTimeStep(self, solver):
		pass

class FE(StepperBase):
	# def __init__(self, dt=0.):
	# 	self.TimeStep = dt
	# 	self.dt = dt

	def TakeTimeStep(self, solver):
		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh = solver.mesh
		U = EqnSet.U

		# Residual, dU arrays
		# try: 
		# 	R = DataSet.R
		# except AttributeError: 
		# 	R = np.copy(U)
		# 	DataSet.R = R
		# try: 
		# 	dU = DataSet.dU
		# except AttributeError: 
		# 	dU = np.copy(U)
		# 	DataSet.dU = dU

		R = self.R 

		R = solver.calculate_residual(U, R)
		dU = mult_inv_mass_matrix(mesh, solver, self.dt, R)
		U += dU

		solver.apply_limiter(U)

		return R


class RK4(StepperBase):
	# def __init__(self, U):
	# 	super().__init__(U)
	# 	self.dU1 = np.zeros_like(U)
	# 	self.dU2 = np.zeros_like(U)
	# 	self.dU3 = np.zeros_like(U)
	# 	self.dU4 = np.zeros_like(U)

	def TakeTimeStep(self, solver):
		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh = solver.mesh
		U = EqnSet.U

		# Residual, dU arrays
		# try: 
		# 	R = DataSet.R
		# except AttributeError: 
		# 	R = np.copy(U)
		# 	DataSet.R = R
		# try: 
		# 	dU = DataSet.dU
		# except AttributeError: 
		# 	dU = np.copy(U)
		# 	DataSet.dU = dU
		# try: 
		# 	dU1 = DataSet.dU1
		# except AttributeError: 
		# 	dU1 = np.copy(U)
		# 	DataSet.dU1 = dU1
		# try: 
		# 	dU2 = DataSet.dU2
		# except AttributeError: 
		# 	dU2 = np.copy(U)
		# 	DataSet.dU2 = dU2
		# try: 
		# 	dU3 = DataSet.dU3
		# except AttributeError: 
		# 	dU3 = np.copy(U)
		# 	DataSet.dU3 = dU3
		# try: 
		# 	dU4 = DataSet.dU4
		# except AttributeError: 
		# 	dU4 = np.copy(U)
		# 	DataSet.dU4 = dU4
		# try: 
		# 	Utemp = DataSet.Utemp
		# except AttributeError: 
		# 	Utemp = np.copy(U)
		# 	DataSet.Utemp = Utemp

		R = self.R
		# dU1 = self.dU1
		# dU2 = self.dU2
		# dU3 = self.dU3
		# dU4 = self.dU4
		
		# print(U[0],U[1])

		# first stage
		R = solver.calculate_residual(U, R)
		dU1 = mult_inv_mass_matrix(mesh, solver, self.dt, R)
		# Utemp.SetToSum(U, dU1, c2=0.5)
		Utemp = U + 0.5*dU1
		solver.apply_limiter(Utemp)
		# second stage
		solver.Time += self.dt/2.
		R = solver.calculate_residual(Utemp, R)
		dU2 = mult_inv_mass_matrix(mesh, solver, self.dt, R)
		Utemp = U + 0.5*dU2
		solver.apply_limiter(Utemp)
		# third stage
		R = solver.calculate_residual(Utemp, R)
		dU3 = mult_inv_mass_matrix(mesh, solver, self.dt, R)
		Utemp = U + dU3
		solver.apply_limiter(Utemp)
		# fourth stage
		solver.Time += self.dt/2.
		R = solver.calculate_residual(Utemp, R)
		dU4 = mult_inv_mass_matrix(mesh, solver, self.dt, R)
		dU = 1./6.*(dU1 + 2.*dU2 + 2.*dU3 + dU4)
		U += dU
		solver.apply_limiter(U)

		# print(U[0],U[1])

		# for egrp in range(mesh.nElemGroup): 
		# 	R[egrp][:] = 1./6.*(dU1[egrp][:]+2.*dU2[egrp][:]+2.*dU3[egrp][:]+dU4[egrp][:])
		# 	U[egrp][:] += R[egrp][:]

		# just return residual from fourth stage
		return R


class LSRK4(StepperBase):
	# Low-storage RK4
	def __init__(self, U):
		super().__init__(U)
		self.rk4a = np.array([            0.0, \
		    -567301805773.0/1357537059087.0, \
		    -2404267990393.0/2016746695238.0, \
		    -3550918686646.0/2091501179385.0, \
		    -1275806237668.0/842570457699.0])
		self.rk4b = np.array([ 1432997174477.0/9575080441755.0, \
		    5161836677717.0/13612068292357.0, \
		    1720146321549.0/2090206949498.0, \
		    3134564353537.0/4481467310338.0, \
		    2277821191437.0/14882151754819.0])
		self.rk4c = np.array([             0.0, \
		    1432997174477.0/9575080441755.0, \
		    2526269341429.0/6820363962896.0, \
		    2006345519317.0/3224310063776.0, \
		    2802321613138.0/2924317926251.0])
		self.nstages = 5
		self.dU = np.zeros_like(U)

	def TakeTimeStep(self, solver):

		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh = solver.mesh
		U = EqnSet.U

		# Residual, dU arrays
		# try: 
		# 	R = DataSet.R
		# except AttributeError: 
		# 	R = np.copy(U)
		# 	DataSet.R = R
		# try: 
		# 	dU = DataSet.dU
		# except AttributeError: 
		# 	dU = np.copy(U)
		# 	DataSet.dU = dU
		# try: 
		# 	dUtemp = DataSet.dUtemp
		# except AttributeError: 
		# 	dUtemp = np.copy(U)
		# 	DataSet.dUtemp = dUtemp

		R = self.R
		dU = self.dU

		Time = solver.Time
		for INTRK in range(self.nstages):
			solver.Time = Time + self.rk4c[INTRK]*self.dt
			R = solver.calculate_residual(U, R)
			dUtemp = mult_inv_mass_matrix(mesh, solver, self.dt, R)
			dU *= self.rk4a[INTRK]
			dU += dUtemp
			U += self.rk4b[INTRK]*dU
			solver.apply_limiter(U)

		return R

class SSPRK3(StepperBase):
	# Low-storage SSPRK3 with 5 stages (as written in Spiteri. 2002)
	def __init__(self, U):
		super().__init__(U)
		self.ssprk3a = np.array([	0.0, \
				-2.60810978953486, \
				-0.08977353434746, \
			-0.60081019321053, \
			-0.72939715170280])
		self.ssprk3b = np.array([ 0.67892607116139, \
			0.20654657933371, \
			0.27959340290485, \
			0.31738259840613, \
			0.30319904778284])
		self.nstages = 5
		self.dU = np.zeros_like(U)

	def TakeTimeStep(self, solver):
		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh = solver.mesh
		U = EqnSet.U

		# Residual, dU arrays
		# try:
		# 	R = DataSet.R
		# except AttributeError:
		# 	R = np.copy(U)
		# 	DataSet.R = R
		# try:
		# 	dU = DataSet.dU
		# except AttributeError:
		# 	dU = np.copy(U)
		# 	DataSet.dU = dU
		# try:	
		# 	dUtemp = DataSet.dUtemp
		# except AttributeError:
		# 	dUtemp = np.copy(U)
		# 	DataSet.dUtemp = dUtemp

		R = self.R
		dU = self.dU

		Time = solver.Time
		for INTRK in range(self.nstages):
			solver.Time = Time + self.dt
			R = solver.calculate_residual(U, R)
			dUtemp = mult_inv_mass_matrix(mesh, solver, self.dt, R)
			dU *= self.ssprk3a[INTRK]
			dU += dUtemp
			U += self.ssprk3b[INTRK]*dU
			solver.apply_limiter(U)
		return R	


class ADER(StepperBase):
	
	def TakeTimeStep(self, solver):
		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh = solver.mesh
		W = EqnSet.U
		Up = EqnSet.Up

		R = self.R

		# Prediction Step (Non-linear Case)
		Up = solver.calculate_predictor_step(self.dt, W, Up)

		# Correction Step
		R = solver.calculate_residual(Up, R)

		dU = mult_inv_mass_matrix(mesh, solver, self.dt/2., R)

		W += dU
		solver.apply_limiter(W)
		return R

class Strang(StepperBase):

	def set_split_schemes(self, explicit, implicit, U):
		param = {"TimeScheme":explicit}
		self.explicit = set_stepper(param, U)

		if ODESolverType[implicit] == ODESolverType.BDF1:
			self.implicit = self.BDF1(U)
		elif ODESolverType[implicit] == ODESolverType.Trapezoidal:
			self.implicit = self.Trapezoidal(U)
		else:
			raise NotImplementedError("Time scheme not supported")

	def TakeTimeStep(self, solver):

		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh  = solver.mesh
		U = EqnSet.U

		explicit = self.explicit
		explicit.dt = self.dt/2.
		implicit = self.implicit
		implicit.dt = self.dt
		#First: take the half-step for the inviscid flux only
		solver.Params["SourceSwitch"] = False
		R1 = explicit.TakeTimeStep(solver)
		#Second: take the implicit full step for the source term.
		solver.Params["SourceSwitch"] = True
		solver.Params["ConvFluxSwitch"] = False

		R2 = implicit.TakeTimeStep(solver)
		# EqnSet.U = U
		#Third: take the second half-step for the inviscid flux only.
		solver.Params["SourceSwitch"] = False
		solver.Params["ConvFluxSwitch"] = True
		R3 = explicit.TakeTimeStep(solver)

		return R3

	class BDF1(StepperBase):

		BETA = 1.0

		def TakeTimeStep(self, solver):

			EqnSet = solver.EqnSet
			DataSet = solver.DataSet
			mesh = solver.mesh
			U = EqnSet.U

			R = self.R

			R = solver.calculate_residual(U, R)
			dU = mult_inv_mass_matrix(mesh, solver, self.dt, R)

			A, iA = self.get_jacobian_matrix(mesh, solver)

			R = np.einsum('ijkl,ikl->ijl',A,U) + dU
			U = np.einsum('ijkl,ikl->ijl',iA,R)

			EqnSet.U = U
			solver.apply_limiter(U)

			return R
		def get_jacobian_matrix(self, mesh, solver):

				basis = solver.basis
				nb = basis.nb
				DataSet = solver.DataSet
				physics = solver.EqnSet
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
			physics = solver.EqnSet
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



