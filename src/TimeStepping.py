import numpy as np 
from Data import ArrayList
from SolverTools import MultInvMassMatrix


class FE(object):
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
			R = ArrayList(SimilarArray=U)
			DataSet.R = R
		try: 
			dU = DataSet.dU
		except AttributeError: 
			dU = ArrayList(SimilarArray=U)
			DataSet.dU = dU

		R = solver.CalculateResidual(U, R)
		MultInvMassMatrix(mesh, solver, self.dt, R, dU)
		U.AddToSelf(dU)

		return R


class RK4(FE):
	def TakeTimeStep(self, solver):
		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh = solver.mesh
		U = EqnSet.U

		# Residual, dU arrays
		try: 
			R = DataSet.R
		except AttributeError: 
			R = ArrayList(SimilarArray=U)
			DataSet.R = R
		try: 
			dU = DataSet.dU
		except AttributeError: 
			dU = ArrayList(SimilarArray=U)
			DataSet.dU = dU
		try: 
			dU1 = DataSet.dU1
		except AttributeError: 
			dU1 = ArrayList(SimilarArray=U)
			DataSet.dU1 = dU1
		try: 
			dU2 = DataSet.dU2
		except AttributeError: 
			dU2 = ArrayList(SimilarArray=U)
			DataSet.dU2 = dU2
		try: 
			dU3 = DataSet.dU3
		except AttributeError: 
			dU3 = ArrayList(SimilarArray=U)
			DataSet.dU3 = dU3
		try: 
			dU4 = DataSet.dU4
		except AttributeError: 
			dU4 = ArrayList(SimilarArray=U)
			DataSet.dU4 = dU4
		try: 
			Utemp = DataSet.Utemp
		except AttributeError: 
			Utemp = ArrayList(SimilarArray=U)
			DataSet.Utemp = Utemp

		# first stage
		R = solver.CalculateResidual(U, R)
		MultInvMassMatrix(mesh, solver, self.dt, R, dU1)
		Utemp.SetToSum(U, dU1, c2=0.5)
		# second stage
		solver.Time += self.dt/2.
		R = solver.CalculateResidual(Utemp, R)
		MultInvMassMatrix(mesh, solver, self.dt, R, dU2)
		Utemp.SetToSum(U, dU2, c2=0.5)
		# third stage
		R = solver.CalculateResidual(Utemp, R)
		MultInvMassMatrix(mesh, solver, self.dt, R, dU3)
		Utemp.SetToSum(U, dU3)
		# fourth stage
		solver.Time += self.dt/2.
		R = solver.CalculateResidual(Utemp, R)
		MultInvMassMatrix(mesh, solver, self.dt, R, dU4)
		dU.SetToSum(dU1, dU2, c2=2.)
		dU.AddToSelf(dU3, c=2.)
		dU.AddToSelf(dU4)
		dU.ScaleByFactor(1./6.)
		U.AddToSelf(dU)
		# for egrp in range(mesh.nElemGroup): 
		# 	R[egrp][:] = 1./6.*(dU1[egrp][:]+2.*dU2[egrp][:]+2.*dU3[egrp][:]+dU4[egrp][:])
		# 	U[egrp][:] += R[egrp][:]

		# just return residual from fourth stage
		return R


class LSRK4(FE):
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
			R = ArrayList(SimilarArray=U)
			DataSet.R = R
		try: 
			dU = DataSet.dU
		except AttributeError: 
			dU = ArrayList(SimilarArray=U)
			DataSet.dU = dU
		try: 
			dUtemp = DataSet.dUtemp
		except AttributeError: 
			dUtemp = ArrayList(SimilarArray=U)
			DataSet.dUtemp = dUtemp

		Time = solver.Time
		for INTRK in range(self.nStage):
			solver.Time = Time + self.rk4c[INTRK]*self.dt
			R = solver.CalculateResidual(U, R)
			MultInvMassMatrix(mesh, solver, self.dt, R, dUtemp)
			dU.ScaleByFactor(self.rk4a[INTRK])
			dU.AddToSelf(dUtemp)
			U.AddToSelf(dU, c=self.rk4b[INTRK])

		return R


