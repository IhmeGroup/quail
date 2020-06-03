import numpy as np 
import code
from abc import ABC, abstractmethod
from Data import ArrayList
from SolverTools import mult_inv_mass_matrix

class StepperBase(ABC):
	def __init__(self, dt=0.):
		self.TimeStep = dt

	@abstractmethod
	def TakeTimeStep(self, solver):
		pass

class FE(StepperBase):
	def __init__(self, dt=0.):
		self.TimeStep = dt

	def TakeTimeStep(self, solver):
		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh = solver.mesh
		U = EqnSet.U

		# Residual, dU arrays
		try: 
			R = DataSet.R
		except AttributeError: 
			R = np.copy(U)
			DataSet.R = R
		try: 
			dU = DataSet.dU
		except AttributeError: 
			dU = np.copy(U)
			DataSet.dU = dU

		R = solver.calculate_residual(U, R)
		mult_inv_mass_matrix(mesh, solver, self.dt, R, dU)
		U += dU

		solver.apply_limiter(U)

		return R


class RK4(StepperBase):
	def TakeTimeStep(self, solver):
		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh = solver.mesh
		U = EqnSet.U

		# Residual, dU arrays
		try: 
			R = DataSet.R
		except AttributeError: 
			R = np.copy(U)
			DataSet.R = R
		try: 
			dU = DataSet.dU
		except AttributeError: 
			dU = np.copy(U)
			DataSet.dU = dU
		try: 
			dU1 = DataSet.dU1
		except AttributeError: 
			dU1 = np.copy(U)
			DataSet.dU1 = dU1
		try: 
			dU2 = DataSet.dU2
		except AttributeError: 
			dU2 = np.copy(U)
			DataSet.dU2 = dU2
		try: 
			dU3 = DataSet.dU3
		except AttributeError: 
			dU3 = np.copy(U)
			DataSet.dU3 = dU3
		try: 
			dU4 = DataSet.dU4
		except AttributeError: 
			dU4 = np.copy(U)
			DataSet.dU4 = dU4
		try: 
			Utemp = DataSet.Utemp
		except AttributeError: 
			Utemp = np.copy(U)
			DataSet.Utemp = Utemp
		# first stage
		R = solver.calculate_residual(U, R)
		mult_inv_mass_matrix(mesh, solver, self.dt, R, dU1)
		# Utemp.SetToSum(U, dU1, c2=0.5)
		Utemp = U + 0.5*dU1
		solver.apply_limiter(Utemp)
		# second stage
		solver.Time += self.dt/2.
		R = solver.calculate_residual(Utemp, R)
		mult_inv_mass_matrix(mesh, solver, self.dt, R, dU2)
		Utemp = U + 0.5*dU2
		solver.apply_limiter(Utemp)
		# third stage
		R = solver.calculate_residual(Utemp, R)
		mult_inv_mass_matrix(mesh, solver, self.dt, R, dU3)
		Utemp = U + dU3
		solver.apply_limiter(Utemp)
		# fourth stage
		solver.Time += self.dt/2.
		R = solver.calculate_residual(Utemp, R)
		mult_inv_mass_matrix(mesh, solver, self.dt, R, dU4)
		dU = 1./6.*(dU1 + 2.*dU2 + 2.*dU3 + dU4)
		U += dU
		solver.apply_limiter(U)
		# for egrp in range(mesh.nElemGroup): 
		# 	R[egrp][:] = 1./6.*(dU1[egrp][:]+2.*dU2[egrp][:]+2.*dU3[egrp][:]+dU4[egrp][:])
		# 	U[egrp][:] += R[egrp][:]

		# just return residual from fourth stage
		return R


class LSRK4(StepperBase):
	# Low-storage RK4
	def __init__(self, dt=0.):
		FE.__init__(self, dt)
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
		self.nStage = 5

	def TakeTimeStep(self, solver):

		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh = solver.mesh
		U = EqnSet.U

		# Residual, dU arrays
		try: 
			R = DataSet.R
		except AttributeError: 
			R = np.copy(U)
			DataSet.R = R
		try: 
			dU = DataSet.dU
		except AttributeError: 
			dU = np.copy(U)
			DataSet.dU = dU
		try: 
			dUtemp = DataSet.dUtemp
		except AttributeError: 
			dUtemp = np.copy(U)
			DataSet.dUtemp = dUtemp

		Time = solver.Time
		for INTRK in range(self.nStage):
			solver.Time = Time + self.rk4c[INTRK]*self.dt
			R = solver.calculate_residual(U, R)
			mult_inv_mass_matrix(mesh, solver, self.dt, R, dUtemp)
			dU *= self.rk4a[INTRK]
			dU += dUtemp
			U += self.rk4b[INTRK]*dU
			solver.apply_limiter(U)

		return R

class SSPRK3(StepperBase):
	# Low-storage SSPRK3 with 5 stages (as written in Spiteri. 2002)
	def __init__(self, dt=0.):
		FE.__init__(self, dt)
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
		self.nStage = 5

	def TakeTimeStep(self, solver):
		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh = solver.mesh
		U = EqnSet.U

		# Residual, dU arrays
		try:
			R = DataSet.R
		except AttributeError:
			R = np.copy(U)
			DataSet.R = R
		try:
			dU = DataSet.dU
		except AttributeError:
			dU = np.copy(U)
			DataSet.dU = dU
		try:	
			dUtemp = DataSet.dUtemp
		except AttributeError:
			dUtemp = np.copy(U)
			DataSet.dUtemp = dUtemp

		Time = solver.Time
		for INTRK in range(self.nStage):
			solver.Time = Time + self.dt
			R = solver.calculate_residual(U, R)
			mult_inv_mass_matrix(mesh, solver, self.dt, R, dUtemp)
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
		try:
			R = DataSet.R
		except AttributeError:
			R = np.copy(W)
			DataSet.R = R
		try: 
			dU = DataSet.dU
		except AttributeError:
			dU = np.copy(W)
			DataSet.dU=dU

		# Prediction Step (Non-linear Case)
		Up = solver.calculate_predictor_step(self.dt, W, Up)

		# Correction Step
		R = solver.calculate_residual(Up, R)

		mult_inv_mass_matrix(mesh, solver, self.dt/2., R, dU)

		W += dU
		solver.apply_limiter(W)
		return R

