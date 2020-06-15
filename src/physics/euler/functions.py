import code
from enum import Enum, auto
import numpy as np
from physics.base.data import FcnBase, BCWeakRiemann, BCWeakPrescribed, SourceBase
from scipy.optimize import fsolve, root


class FcnType(Enum):
    SmoothIsentropicFlow = auto()
    MovingShock = auto()
    IsentropicVortex = auto()
    DensityWave = auto()

class SourceType(Enum):
    StiffFriction = auto()

'''
State functions
'''
class smooth_isentropic_flow(FcnBase):
	def __init__(self, a=0.9):
		self.a = a

	def get_state(self, physics, x, t):
		
		a = self.a
		gam = physics.Params["SpecificHeatRatio"]
		irho, irhou, irhoE = physics.GetStateIndices()
	
		# Up = np.zeros([x.shape[0], physics.StateRank])

		rho0 = lambda x,a: 1. + a*np.sin(np.pi*x)
		pressure = lambda rho,gam: rho**gam
		rho = lambda x1,x2,a: 0.5*(rho0(x1,a) + rho0(x2,a))
		vel = lambda x1,x2,a: np.sqrt(3)*(rho(x1,x2,a) - rho0(x1,a))

		f1 = lambda x1,x,t,a: x + np.sqrt(3)*rho0(x1,a)*t - x1
		f2 = lambda x2,x,t,a: x - np.sqrt(3)*rho0(x2,a)*t - x2

		x_ = x.reshape(-1)

		if isinstance(t,float):
			Up = np.zeros([x.shape[0], physics.StateRank])

			x1 = fsolve(f1, 0.*x_, (x_,t,a))
			if np.abs(x1.any()) > 1.: raise Exception("x1 = %g out of range" % (x1))
			x2 = fsolve(f2, 0.*x_, (x_,t,a))
			if np.abs(x2.any()) > 1.: raise Exception("x2 = %g out of range" % (x2))
		else:

			Up = np.zeros([t.shape[0], physics.StateRank])

			y = np.zeros(len(t))
			for i in range(len(t)):
			#	code.interact(local=locals())
				y[i] = x
			#y = x.transpose()
			y_ = y.reshape(-1)
			t = t.reshape(-1)

			x1 = root(f1, 0.*y_, (y_,t,a)).x
			if np.abs(x1.any()) > 1.: raise Exception("x1 = %g out of range" % (x1))
			x2 = root(f2, 0.*y_, (y_,t,a)).x
			if np.abs(x2.any()) > 1.: raise Exception("x2 = %g out of range" % (x2))
			
		r = rho(x1,x2,a)
		u = vel(x1,x2,a)
		p = pressure(r,gam)
		rE = p/(gam-1.) + 0.5*r*u*u

		Up[:,irho] = r
		Up[:,irhou] = r*u
		Up[:,irhoE] = rE

		return Up

class moving_shock(FcnBase):
	def __init__(self, M = 5.0, xshock = 0.2):
		self.M = M
		self.xshock = xshock

	def get_state(self, physics, x, t):

		M = self.M
		xshock = self.xshock
		irho = physics.GetStateIndex("Density")
		irhou = physics.GetStateIndex("XMomentum")
		irhoE = physics.GetStateIndex("Energy")
		
		# Up = np.zeros([x.shape[0], physics.StateRank])
		if physics.Dim == 2: irhov = physics.GetStateIndex("YMomentum")
		gam = physics.Params["SpecificHeatRatio"]
		
		if not isinstance(t,float):
			Up = np.zeros([t.shape[0], physics.StateRank])

			t = t.reshape(-1)
			y = np.zeros(len(t))
			for i in range(len(t)):
				y[i]=x
			x = y

			rho1 = np.full(len(t),1.)
			p1 = np.full(len(t),1.e5)
			u1 = np.full(len(t),0.)
		else:
			Up = np.zeros([x.shape[0], physics.StateRank])

			''' Pre-shock state '''
			rho1 = 1.
			p1 = 1.e5
			u1 = 0.

		''' Update xshock based on shock speed '''
		a1 = np.sqrt(gam*p1/rho1)
		W = M*a1 
		us = u1 + W # shock speed in lab frame
		xshock = xshock + us*t

		''' Post-shock state '''
		rho2 = (gam+1.)*M**2./((gam-1.)*M**2. + 2.)*rho1
		p2 = (2.*gam*M**2. - (gam-1.))/(gam + 1.)*p1
		# To get velocity, first work in reference frame fixed to shock
		ux = W
		uy = ux*rho1/rho2
		# Convert back to lab frame
		u2 = W + u1 - uy

		''' Fill state '''
		ileft = (x <= xshock).reshape(-1)
		iright = (x > xshock).reshape(-1)
		if not isinstance(t,float):
			for i in range(len(t)):
				# Density
				Up[iright[i], i, irho] = rho1[i]
				Up[ileft[i], i, irho] = rho2[i]
				# Momentum
				Up[iright[i], i, irhou] = rho1[i]*u1[i]
				Up[ileft[i], i, irhou] = rho2[i]*u2[i]
				if physics.Dim == 2: Up[:, irhov] = 0.
				# Energy
				Up[iright[i], i, irhoE] = p1[i]/(gam-1.) + 0.5*rho1[i]*u1[i]*u1[i]
				Up[ileft[i], i, irhoE] = p2[i]/(gam-1.) + 0.5*rho2[i]*u2[i]*u2[i]

		else:
			# Density
			Up[iright, irho] = rho1
			Up[ileft, irho] = rho2
			# Momentum
			Up[iright, irhou] = rho1*u1
			Up[ileft, irhou] = rho2*u2
			if physics.Dim == 2: Up[:, irhov] = 0.
			# Energy
			Up[iright, irhoE] = p1/(gam-1.) + 0.5*rho1*u1*u1
			Up[ileft, irhoE] = p2/(gam-1.) + 0.5*rho2*u2*u2

		return Up

class density_wave(FcnBase):
	def __init__(self, p = 1.0):
		self.p = p

	def get_state(self, physics, x, t):
		p = self.p
		irho, irhou, irhoE = physics.GetStateIndices()
		gam = physics.Params["SpecificHeatRatio"]

		Up = np.zeros([x.shape[0], physics.StateRank])

		x_ = x.reshape(-1)
		
		r = 1.0+0.1*np.sin(2.*np.pi*x_)
		ru = r*1.0
		rE = (p/(gam-1.))+0.5*ru**2/r

		Up[:,irho] = r
		Up[:,irhou] = ru
		Up[:,irhoE] = rE

		return Up
class isentropic_vortex(FcnBase):
	def __init__(self,rhob=1.,ub=1.,vb=1.,pb=1.,vs=5.):
		self.rhob = 1.
		self.ub = 1.
		self.vb = 1.
		self.pb = 1.
		self.vs = 5.
	def get_state(self,physics,x,t):		
		Up = np.zeros([x.shape[0], physics.StateRank])
		gam = physics.Params["SpecificHeatRatio"]
		Rg = physics.Params["GasConstant"]

		### Parameters
		# Base flow
		rhob = self.rhob
		# x-velocity
		ub = self.ub
		# y-velocity
		vb = self.vb
		# pressure
		pb = self.pb
		# vortex strength
		vs = self.vs
		# Make sure Rg is 1
		if Rg != 1.:
			raise ValueError

		# Base temperature
		Tb = pb/(rhob*Rg)

		# Entropy
		s = pb/rhob**gam

		xr = x[:,0] - ub*t
		yr = x[:,1] - vb*t
		r = np.sqrt(xr**2. + yr**2.)

		# Perturbations
		dU = vs/(2.*np.pi)*np.exp(0.5*(1-r**2.))
		du = dU*-yr
		dv = dU*xr

		dT = -(gam - 1.)*vs**2./(8.*gam*np.pi**2.)*np.exp(1.-r**2.)

		u = ub + du 
		v = vb + dv 
		T = Tb + dT

		# Convert to conservative variables
		r = np.power(T/s, 1./(gam-1.))
		ru = r*u
		rv = r*v
		rE = r*Rg/(gam-1.)*T + 0.5*(ru*ru + rv*rv)/r

		Up[:,0] = r
		Up[:,1] = ru
		Up[:,2] = rv
		Up[:,3] = rE

		return Up


'''
Source term functions
'''
class stiff_friction(SourceBase):
	def __init__(self, nu=-1):
		self.nu = nu

	def get_source(self, physics, FcnData, x, t):
		nu = self.nu
		irho = physics.GetStateIndex("Density")
		irhou = physics.GetStateIndex("XMomentum")
		irhoE = physics.GetStateIndex("Energy")
		
		U = FcnData.U
		
		S = np.zeros_like(U)

		eps = 1.0e-12
		S[:,irho] = 0.0
		S[:,irhou] = nu*(U[:,irhou])
		S[:,irhoE] = nu*((U[:,irhou])**2/(eps+U[:,irho]))
		
		return S

	def get_jacobian(self, U):

		nu = self.nu
		
		jac = np.zeros([U.shape[-1],U.shape[-1]])
		vel = U[:,1]/(1.0e-12+U[:,0])

		jac[1,1]=nu
		jac[2,0]=-nu*vel**2
		jac[2,1]=2.0*nu*vel

		return jac



