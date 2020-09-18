import code
from enum import Enum, auto
import numpy as np
from scipy.optimize import fsolve, root

from physics.base.data import FcnBase, BCWeakRiemann, BCWeakPrescribed, SourceBase, ConvNumFluxBase


class FcnType(Enum):
    SmoothIsentropicFlow = auto()
    MovingShock = auto()
    IsentropicVortex = auto()
    DensityWave = auto()
    RiemannProblem = auto()
    # SmoothRiemannProblem = auto()
    TaylorGreenVortex = auto()
    ExactRiemannSolution = auto()

class BCType(Enum):
	SlipWall = auto()
	PressureOutlet = auto()


class SourceType(Enum):
    StiffFriction = auto()
    TaylorGreenSource = auto()


class ConvNumFluxType(Enum):
	Roe = auto()
	HLLC = auto()


'''
State functions
'''

class SmoothIsentropicFlow(FcnBase):
	def __init__(self, a=0.9):
		self.a = a

	def get_state(self, physics, x, t):
		
		a = self.a
		gamma = physics.gamma
		irho, irhou, irhoE = physics.get_state_indices()
	
		# Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])

		rho0 = lambda x, a: 1. + a*np.sin(np.pi*x)
		pressure = lambda rho, gamma: rho**gamma
		rho = lambda x1, x2, a: 0.5*(rho0(x1, a) + rho0(x2, a))
		vel = lambda x1, x2, a: np.sqrt(3)*(rho(x1, x2, a) - rho0(x1, a))

		f1 = lambda x1, x, t, a: x + np.sqrt(3)*rho0(x1, a)*t - x1
		f2 = lambda x2, x, t, a: x - np.sqrt(3)*rho0(x2, a)*t - x2

		xr = x.reshape(-1)

		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])

		x1 = fsolve(f1, 0.*xr, (xr, t, a))
		if np.abs(x1.any()) > 1.: raise Exception("x1 = %g out of range" % (x1))
		x2 = fsolve(f2, 0.*xr, (xr, t, a))
		if np.abs(x2.any()) > 1.: raise Exception("x2 = %g out of range" % (x2))
			
		den = rho(x1, x2, a)
		u = vel(x1, x2, a)
		p = pressure(den, gamma)
		rhoE = p/(gamma - 1.) + 0.5*den*u*u

		Uq[:, irho] = den
		Uq[:, irhou] = den*u
		Uq[:, irhoE] = rhoE

		return Uq


class MovingShock(FcnBase):
	def __init__(self, M = 5.0, xshock = 0.2):
		self.M = M
		self.xshock = xshock

	def get_state(self, physics, x, t):

		M = self.M
		xshock = self.xshock

		srho, srhou, srhoE = physics.get_state_slices()

		gamma = physics.gamma
		
		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])
		''' Pre-shock state '''
		rho1 = 1.
		p1 = 1.e5
		u1 = 0.

		''' Update xshock based on shock speed '''
		a1 = np.sqrt(gamma*p1/rho1)
		W = M*a1 
		us = u1 + W # shock speed in lab frame
		xshock = xshock + us*t

		''' Post-shock state '''
		rho2 = (gamma + 1.)*M**2./((gamma - 1.)*M**2. + 2.)*rho1
		p2 = (2.*gamma*M**2. - (gamma - 1.))/(gamma + 1.)*p1
		# To get velocity, first work in reference frame fixed to shock
		ux = W
		uy = ux*rho1/rho2
		# Convert back to lab frame
		u2 = W + u1 - uy

		''' Fill state '''
		ileft = (x <= xshock).reshape(-1)
		iright = (x > xshock).reshape(-1)

		# Density
		Uq[iright, srho] = rho1
		Uq[ileft, srho] = rho2
		# Momentum
		Uq[iright, srhou] = rho1*u1
		Uq[ileft, srhou] = rho2*u2
		# Energy
		Uq[iright, srhoE] = p1/(gamma - 1.) + 0.5*rho1*u1*u1
		Uq[ileft, srhoE] = p2/(gamma - 1.) + 0.5*rho2*u2*u2

		return Uq


class IsentropicVortex(FcnBase):
	def __init__(self, rhob=1., ub=1., vb=1., pb=1., vs=5.):
		self.rhob = rhob
		self.ub = ub
		self.vb = vb
		self.pb = pb
		self.vs = vs

	def get_state(self, physics, x, t):		
		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])
		gamma = physics.gamma
		Rg = physics.R

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
		s = pb/rhob**gamma

		xr = x[:,0] - ub*t
		yr = x[:,1] - vb*t
		r = np.sqrt(xr**2. + yr**2.)

		# Perturbations
		dU = vs/(2.*np.pi)*np.exp(0.5*(1-r**2.))
		du = dU*-yr
		dv = dU*xr

		dT = -(gamma - 1.)*vs**2./(8.*gamma*np.pi**2.)*np.exp(1. - r**2.)

		u = ub + du 
		v = vb + dv 
		T = Tb + dT

		# Convert to conservative variables
		rho = np.power(T/s, 1./(gamma - 1.))
		rhou = rho*u
		rhov = rho*v
		rhoE = rho*Rg/(gamma - 1.)*T + 0.5*(rhou*rhou + rhov*rhov)/rho

		Uq[:, 0] = rho
		Uq[:, 1] = rhou
		Uq[:, 2] = rhov
		Uq[:, 3] = rhoE

		return Uq


class DensityWave(FcnBase):
	def __init__(self, p=1.0):
		self.p = p

	def get_state(self, physics, x, t):
		p = self.p
		srho, srhou, srhoE = physics.get_state_slices()
		gamma = physics.gamma

		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])
		
		rho = 1.0 + 0.1*np.sin(2.*np.pi*x)
		rhou = rho*1.0
		rhoE = (p/(gamma - 1.)) + 0.5*rhou**2/rho

		Uq[:, srho] = rho
		Uq[:, srhou] = rhou
		Uq[:, srhoE] = rhoE

		return Uq


# class RiemannProblem(FcnBase):
# 	def __init__(self, uL=np.array([1., 0., 1.]), 
# 			uR=np.array([0.125, 0., 0.1]), xshock=0.):
# 		# Default conditions set up for Sod Problem.
# 		self.uL = uL
# 		self.uR = uR
# 		self.xshock = xshock

# 	def get_state(self, physics, x, t):

# 		xshock = self.xshock
# 		uL = self.uL
# 		uR = self.uR

# 		rhoL = uL[0]
# 		vL = uL[1]
# 		pL = uL[2]
		
# 		rhoR = uR[0]
# 		vR = uR[1]
# 		pR = uR[2]

# 		srho, srhou, srhoE = physics.get_state_slices()

# 		gamma = physics.gamma
		
# 		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])

# 		''' Fill state '''
# 		ileft = (x <= xshock).reshape(-1)
# 		iright = (x > xshock).reshape(-1)

# 		# Density
# 		Uq[iright, srho] = rhoR
# 		Uq[ileft, srho] = rhoL
# 		# Momentum
# 		Uq[iright, srhou] = rhoR*vR
# 		Uq[ileft, srhou] = rhoL*vL
# 		# Energy
# 		Uq[iright, srhoE] = pR/(gamma-1.) + 0.5*rhoR*vR*vR
# 		Uq[ileft, srhoE] = pL/(gamma-1.) + 0.5*rhoL*vL*vL

# 		return Uq


class ExactRiemannSolution(FcnBase):
	def __init__(self, rhoL=1., uL=0., pL=1., rhoR=0.125, uR=0., pR=0.1,
				xshock=0.):
		# Default conditions set up for Sod Problem.
		self.rhoL = rhoL
		self.uL = uL
		self.pL = pL
		self.rhoR = rhoR
		self.uR = uR
		self.pR = pR
		self.xshock = xshock

	def get_state(self, physics, x, t):

		# uL = self.uL
		# uR = self.uR
		xshock = self.xshock
		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])
		gamma = physics.gamma
		srho, srhou, srhoE = physics.get_state_slices()

		# rho4 = uL[0]; p4 = uL[2]; u4 = uL[1]
		# rho1 = uR[0]; p1 = uR[2]; u1 = uR[1]

		rho4 = self.rhoL; p4 = self.pL; u4 = self.uL
		rho1 = self.rhoR; p1 = self.pR; u1 = self.uR

		c4 = np.sqrt(gamma*p4/rho4)
		c1 = np.sqrt(gamma*p1/rho1)
		p41 = p4/p1

		def F(y):
			F = y * (1. + (gamma-1.)/(2.*c4) * (u4 - u1 - c1/gamma*(y-1.)/ \
					np.sqrt((gamma+1.)/(2.*gamma)*(y-1.) + 1)))**(-2. \
					*gamma/(gamma-1)) - p4/p1
			return F			

		y0 = 0.5*p4/p1
		Y = fsolve(F, y0)

		# can now get p2
		p2 = Y*p1

		# Equation 11
		u2 = u1 + c1/gamma*(p2/p1-1)/np.sqrt((gamma+1)/(2*gamma)*(p2/p1-1) \
				+ 1)
		# Equation 10
		num = (gamma+1)/(gamma-1) + p2/p1
		den = 1 + (gamma+1)/(gamma-1)*(p2/p1)
		c2 = c1*np.sqrt(p2/p1*num/den)
		# Equation 12 - shock speed
		V = u1 + c1*np.sqrt((gamma+1)/(2*gamma)*(p2/p1-1) + 1)
		# density for state 2
		rho2 = gamma*p2/c2**2

		# Equations 13 and 14
		u3 = u2
		p3 = p2 
		# Equation 16
		c3 = (gamma-1)/2*(u4-u3+2/(gamma-1)*c4)
		rho3 = gamma*p3/c3**2

		# now deal with expansion fan
		xe1 = (u4-c4)*t + xshock; # "start" of expansion fan
		xe2 = (t*((gamma+1)/2*u3 - (gamma-1)/2*u4 - c4)+xshock) # end

		# location of shock
		xs = V*t + xshock
		# location of contact
		xc = u2*t + xshock

		uu = np.zeros_like(x); pp = np.zeros_like(x); rr = np.zeros_like(x);

		# for i in range(len(x)):
		for i in range(x.shape[0]):
		    if x[i] <= xe1:
		        uu[i] = u4; pp[i] = p4; rr[i] = rho4;
		    elif x[i] > xe1 and x[i] <= xe2:
		        uu[i] = (2/(gamma+1)*((x[i]-xshock)/t + (gamma-1)/2*u4 + c4)) 
		        cc = uu[i] - (x[i]-xshock)/t
		        pp[i] = p4*(cc/c4)**(2*gamma/(gamma-1))
		        rr[i] = gamma*pp[i]/cc**2
		    elif x[i] > xe2 and x[i] <= xc:
		        uu[i] = u3; pp[i] = p3; rr[i] = rho3;
		    elif x[i] > xc and x[i] <= xs:
		        uu[i] = u2; pp[i] = p2; rr[i] = rho2;
		    else:
		        uu[i] = u1; pp[i] = p1; rr[i] = rho1;

		Uq[:, srho] = rr
		Uq[:, srhou] = rr*uu
		Uq[:, srhoE] = pp/(gamma-1.) + 0.5*rr*uu*uu

		return Uq


class RiemannProblem(FcnBase):
	def __init__(self, rhoL=1., uL=0., pL=1., rhoR=0.125, uR=0., pR=0.1,
				xshock=0., w=1.e-30):
		# Default conditions set up for Sod Problem.
		self.rhoL = rhoL
		self.uL = uL
		self.pL = pL
		self.rhoR = rhoR
		self.uR = uR
		self.pR = pR
		self.xshock = xshock
		self.w = w

	def get_state(self, physics, x, t):

		rhoL = self.rhoL
		uL = self.uL
		pL = self.pL
		rhoR = self.rhoR
		uR = self.uR
		pR = self.pR
		xshock = self.xshock
		# uL = self.uL
		# uR = self.uR
		w = self.w

		# rhoL = uL[0]
		# vL = uL[1]
		# pL = uL[2]
		
		# rhoR = uR[0]
		# vR = uR[1]
		# pR = uR[2]

		srho, srhou, srhoE = physics.get_state_slices()

		gamma = physics.gamma
		
		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])

		# w = 0.05
		def set_tanh(a, b, w, xo):
			return 0.5*((a+b) + (b-a)*np.tanh((x-xo)/w))
		# Density
		Uq[:, srho] =  set_tanh(rhoL, rhoR, w, xshock)

		# Momentum
		Uq[:, srhou] = set_tanh(rhoL*uL, rhoR*uR, w, xshock)
		# Energy
		rhoeL = pL/(gamma-1.) + 0.5*rhoL*uL*uL
		rhoeR = pR/(gamma-1.) + 0.5*rhoR*uR*uR
		Uq[:, srhoE] = set_tanh(rhoeL, rhoeR, w, xshock)

		return Uq


class TaylorGreenVortex(FcnBase):

	def get_state(self, physics, x, t):		
		Uq = np.zeros([x.shape[0], physics.NUM_STATE_VARS])
		gamma = physics.gamma
		Rg = physics.R

		irho, irhou, irhov, irhoE = physics.get_state_indices()

		rho = 1.
		u = np.sin(np.pi*x[:, 0])*np.cos(np.pi*x[:, 1])
		v = -np.cos(np.pi*x[:, 0])*np.sin(np.pi*x[:, 1])
		p = 0.25*(np.cos(2.*np.pi*x[:, 0]) + np.cos(2*np.pi*x[:, 1])) + 1.
		E = p/(rho*(gamma - 1.)) + 0.5*(u**2. + v**2.)

		Uq[:, irho] = rho
		Uq[:, irhou] = rho*u
		Uq[:, irhov] = rho*v
		Uq[:, irhoE] = rho*E

		return Uq


'''
Boundary conditions
'''

class SlipWall(BCWeakPrescribed):
	def get_boundary_state(self, physics, UqI, normals, x, t):
		smom = physics.get_momentum_slice()

		n_hat = normals/np.linalg.norm(normals, axis=1, keepdims=True)

		rhoveln = np.sum(UqI[:, smom] * n_hat, axis=1, keepdims=True)
		UqB = UqI.copy()
		UqB[:, smom] -= rhoveln * n_hat

		return UqB


class PressureOutlet(BCWeakPrescribed):
	def __init__(self, p):
		self.p = p

	def get_boundary_state(self, physics, UqI, normals, x, t):
		srho = physics.get_state_slice("Density")
		srhoE = physics.get_state_slice("Energy")
		smom = physics.get_momentum_slice()

		UqB = UqI.copy()

		n_hat = normals/np.linalg.norm(normals, axis=1, keepdims=True)

		# Pressure
		pB = self.p

		gamma = physics.gamma

		# gamma = physics.gamma
		# igam = 1./gamma
		# gmi = gamma - 1.
		# igmi = 1./gmi

		# Interior velocity in normal direction
		rhoI = UqI[:, srho]
		velI = UqI[:, smom]/rhoI
		velnI = np.sum(velI*n_hat, axis=1, keepdims=True)

		if np.any(velnI < 0.):
			print("Incoming flow at outlet")

		# Compute interior pressure
		# rVI2 = np.sum(UqI[:,imom]**2., axis=1, keepdims=True)/rhoI
		# pI = gmi*(UqI[:,irhoE:irhoE+1] - 0.5*rVI2)
		pI = physics.compute_variable("Pressure", UqI)

		if np.any(pI < 0.):
			raise errors.NotPhysicalError

		# Interior speed of sound
		# cI = np.sqrt(gamma*pI/rhoI)
		cI = physics.compute_variable("SoundSpeed", UqI)
		JI = velnI + 2.*cI/(gamma - 1.)
		veltI = velI - velnI*n_hat

		# Normal Mach number
		Mn = velnI/cI
		if np.any(Mn >= 1.):
			return UqB

		# Boundary density from interior entropy
		rhoB = rhoI*np.power(pB/pI, 1./gamma)
		UqB[:, srho] = rhoB

		# Exterior speed of sound
		cB = np.sqrt(gamma*pB/rhoB)
		velB = (JI - 2.*cB/(gamma-1.))*n_hat + veltI
		UqB[:, smom] = rhoB*velB
		# dVn = 2.*igmi*(cI-cB)
		# UqB[:,imom] = rhoB*dVn*n_hat + rhoB*UqI[:,imom]/rhoI

		# Exterior energy
		# rVB2 = np.sum(UqB[:,imom]**2., axis=1, keepdims=True)/rhoB
		rhovel2B = rhoB*np.sum(velB**2., axis=1, keepdims=True)
		UqB[:, srhoE] = pB/(gamma - 1.) + 0.5*rhovel2B

		return UqB


'''
Source term functions
'''

class StiffFriction(SourceBase):
	def __init__(self, nu=-1):
		self.nu = nu

	def get_source(self, physics, Uq, x, t):
		nu = self.nu
		# irho = physics.get_state_index("Density")
		# irhou = physics.get_state_index("XMomentum")
		# irhoE = physics.get_state_index("Energy")

		irho, irhou, irhoE = physics.get_state_indices()
		
		# U = self.U
		
		S = np.zeros_like(Uq)

		eps = 1.0e-12
		S[:, irho] = 0.0
		S[:, irhou] = nu*(Uq[:, irhou])
		S[:, irhoE] = nu*((Uq[:, irhou])**2/(eps+Uq[:, irho]))
		
		return S

	# def get_jacobian(self, physics, FcnData, x, t):

	# 	nu = self.nu

	# 	Uq = FcnData.Uq
	# 	irho, irhou, irhoE = physics.get_state_indices()

	# 	jac = np.zeros([Uq.shape[0], Uq.shape[-1], Uq.shape[-1]])
	# 	vel = Uq[:, 1]/(1.0e-12 + Uq[:, 0])

	# 	jac[:, irhou, irhou] = nu
	# 	jac[:, irhoE, irho] = -nu*vel**2
	# 	jac[:, irhoE, irhou] = 2.0*nu*vel
	# 	# jac[:, 1,1] = nu
	# 	# jac[:, 2, 0] = -nu*vel**2
	# 	# jac[:, 2, 1] = 2.0*nu*vel

	# 	return jac
	def get_jacobian(self, physics, Uq, x, t):

		nu = self.nu
		# Uq = self.Uq

		irho, irhou, irhoE = physics.get_state_indices()

		jac = np.zeros([Uq.shape[0], Uq.shape[-1], Uq.shape[-1]])
		vel = Uq[:, 1]/(1.0e-12 + Uq[:, 0])

		jac[:, irhou, irhou] = nu
		jac[:, irhoE, irho] = -nu*vel**2
		jac[:, irhoE, irhou] = 2.0*nu*vel
		# jac[:, 1, 1] = nu
		# jac[:, 2, 0] = -nu*vel**2
		# jac[:, 2, 1] = 2.0*nu*vel

		return jac


class TaylorGreenSource(SourceBase):

	def get_source(self, physics, Uq, x, t):
		gamma = physics.gamma

		irho, irhou, irhov, irhoE = physics.get_state_indices()
		
		# Uq = self.Uq
		
		S = np.zeros_like(Uq)

		S[:, irhoE] = np.pi/(4.*(gamma - 1.))*(np.cos(3.*np.pi*x[:, 0])* \
				np.cos(np.pi*x[:, 1]) - np.cos(np.pi*x[:, 0])*np.cos(3.* \
				np.pi*x[:, 1]))
		
		return S


'''
Numerical flux functions
'''

class Roe1D(ConvNumFluxBase):
	def __init__(self, Uq=None):
		if Uq is not None:
			n = Uq.shape[0]
			ns = Uq.shape[1]
			dim = ns - 2
		else:
			n = 0; ns = 0; dim = 0

		# self.velL = np.zeros([n,dim])
		# self.velR = np.zeros([n,dim])
		self.UqL = np.zeros_like(Uq)
		self.UqR = np.zeros_like(Uq)
		self.vel = np.zeros([n, dim])
		# self.rhoL_sqrt = np.zeros([n,1])
		# self.rhoR_sqrt = np.zeros([n,1])
		# self.HL = np.zeros([n,1])
		# self.HR = np.zeros([n,1])
		# self.rhoRoe = np.zeros([n,1])
		# self.velRoe = np.zeros([n,dim])
		# self.HRoe = np.zeros([n,1])
		# self.c2 = np.zeros([n,1])
		# self.c = np.zeros([n,1])
		# self.dvel = np.zeros([n,dim])
		# self.drho = np.zeros([n,1])
		# self.dp = np.zeros([n,1])
		self.alphas = np.zeros_like(Uq)
		self.evals = np.zeros_like(Uq)
		self.R = np.zeros([n, ns, ns])
		# self.FRoe = np.zeros_like(u)
		# self.FL = np.zeros_like(u)
		# self.FR = np.zeros_like(u)

	# def AllocHelperArrays(self, u):
	# 	self.__init__(u)

	def rotate_coord_sys(self, smom, Uq, n):
		Uq[:, smom] *= n

		return Uq

	def undo_rotate_coord_sys(self, smom, Uq, n):
		Uq[:, smom] /= n

		return Uq

	def RoeAverageState(self, physics, srho, velL, velR, uL, uR):
		# rhoL_sqrt = self.rhoL_sqrt
		# rhoR_sqrt = self.rhoR_sqrt
		# HL = self.HL 
		# HR = self.HR 

		rhoL_sqrt = np.sqrt(uL[:,srho])
		rhoR_sqrt = np.sqrt(uR[:,srho])
		HL = physics.compute_variable("TotalEnthalpy", uL)
		HR = physics.compute_variable("TotalEnthalpy", uR)

		# self.velRoe = (rhoL_sqrt*velL + rhoR_sqrt*velR)/(rhoL_sqrt+rhoR_sqrt)
		# self.HRoe = (rhoL_sqrt*HL + rhoR_sqrt*HR)/(rhoL_sqrt+rhoR_sqrt)
		# self.rhoRoe = rhoL_sqrt*rhoR_sqrt

		velRoe = (rhoL_sqrt*velL + rhoR_sqrt*velR)/(rhoL_sqrt+rhoR_sqrt)
		HRoe = (rhoL_sqrt*HL + rhoR_sqrt*HR)/(rhoL_sqrt+rhoR_sqrt)
		rhoRoe = rhoL_sqrt*rhoR_sqrt

		return rhoRoe, velRoe, HRoe

	def GetDifferences(self, physics, srho, velL, velR, uL, uR):
		# dvel = self.dvel
		# drho = self.drho
		# dp = self.dp 

		dvel = velR - velL
		drho = uR[:,srho] - uL[:,srho]
		dp = physics.compute_variable("Pressure", uR) - \
			physics.compute_variable("Pressure", uL)

		return dvel, drho, dp

	def get_alphas(self, c, c2, dp, dvel, drho, rhoRoe):
		alphas = self.alphas 

		alphas[:,0:1] = 0.5/c2*(dp - c*rhoRoe*dvel[:,0:1])
		alphas[:,1:2] = drho - dp/c2 
		alphas[:,-1:] = 0.5/c2*(dp + c*rhoRoe*dvel[:,0:1])

		return alphas 

	def get_eigenvalues(self, velRoe, c):
		evals = self.evals 

		evals[:,0:1] = velRoe[:,0:1] - c
		evals[:,1:2] = velRoe[:,0:1]
		evals[:,-1:] = velRoe[:,0:1] + c
		
		# entropy fix
		# ep = 1e-2
		# for iq in range(evals.shape[0]):
		# 	eps = ep*c[iq]
		# 	for i in range(evals.shape[-1]):
		# 		if evals[iq,i] < eps and evals[iq,i] > -eps:
		# 			eps1 = 1./eps
		# 			evals[iq,i] = 0.5*(eps+evals[iq,i]*evals[iq,i]*eps1)
		return evals 

	def get_right_eigenvectors(self, c, evals, velRoe, HRoe):
		R = self.R

		# first row
		# R[:,0,[0,1,-1]] = 1.
		R[:,0,0:2] = 1.; R[:,0,-1] = 1.
		# second row
		R[:,1,0] = evals[:,0]; R[:,1,1] = velRoe[:,0]; R[:,1,-1] = evals[:,-1]
		# last row
		R[:,-1,0:1] = HRoe - velRoe[:,0:1]*c; R[:,-1,1:2] = 0.5*np.sum(velRoe*velRoe, axis=1, keepdims=True)
		R[:,-1,-1:] = HRoe + velRoe[:,0:1]*c

		return R 


	def compute_flux(self, physics, UqL_std, UqR_std, n):
		'''
		Function: ConvFluxLaxFriedrichs
		-------------------
		This function computes the numerical flux (dotted with the normal)
		using the Lax-Friedrichs flux function

		Inputs:
		    gamma: specific heat ratio
		    UqL: Left state
		    UqR: Right state
		    n: Normal vector (assumed left to right)

		Outputs:
		    F: Numerical flux dotted with the normal, i.e., F_hat dot n
		'''

		# Extract helper arrays
		UqL = self.UqL 
		UqR = self.UqR

		# velL = self.velL
		# velR = self.velR 
		# c2 = self.c2
		# c = self.c 
		# alphas = self.alphas 
		# evals = self.evals 
		# R = self.R 
		# FRoe = self.FRoe 
		# FL = self.FL 
		# FR = self.FR 

		# Indices
		srho = physics.get_state_slice("Density")
		smom = physics.get_momentum_slice()

		gamma = physics.gamma

		NN = np.linalg.norm(n, axis=1, keepdims=True)
		n1 = n/NN

		# Copy values before rotating
		UqL[:] = UqL_std
		UqR[:] = UqR_std

		# Rotated coordinate system
		UqL = self.rotate_coord_sys(smom, UqL, n1)
		UqR = self.rotate_coord_sys(smom, UqR, n1)

		# Velocities
		velL = UqL[:, smom]/UqL[:, srho]
		velR = UqR[:, smom]/UqR[:, srho]

		rhoRoe, velRoe, HRoe = self.RoeAverageState(physics, srho, velL, 
				velR, UqL, UqR)

		# Speed of sound from Roe-averaged state
		c2 = (gamma - 1.)*(HRoe - 0.5*np.sum(velRoe*velRoe, axis=1, 
				keepdims=True))
		if np.any(c2 <= 0.):
			raise errors.NotPhysicalError
		c = np.sqrt(c2)

		# differences
		dvel, drho, dp = self.GetDifferences(physics, srho, velL, velR, UqL, UqR)

		# alphas (left eigenvectors multipled by dU)
		# alphas[:,[0]] = 0.5/c2*(dp - c*rhoRoe*dvel[:,[0]])
		# alphas[:,[1]] = drho - dp/c2 
		# alphas[:,ydim] = rhoRoe*dvel[:,[-1]]
		# alphas[:,[-1]] = 0.5/c2*(dp + c*rhoRoe*dvel[:,[0]])
		alphas = self.get_alphas(c, c2, dp, dvel, drho, rhoRoe)

		# Eigenvalues
		# evals[:,[0]] = velRoe[:,[0]] - c
		# evals[:,1:-1] = velRoe[:,[0]]
		# evals[:,[-1]] = velRoe[:,[0]] + c
		evals = self.get_eigenvalues(velRoe, c)

		# Right eigenvector matrix
		# first row
		# R[:,0,[0,1,-1]] = 1.; R[:,0,ydim] = 0.
		# # second row
		# R[:,1,0] = evals[:,0]; R[:,1,1] = velRoe[:,0]; R[:,1,ydim] = 0.; R[:,1,-1] = evals[:,-1]
		# # last row
		# R[:,-1,[0]] = HRoe - velRoe[:,[0]]*c; R[:,-1,[1]] = 0.5*np.sum(velRoe*velRoe, axis=1, keepdims=True)
		# R[:,-1,[-1]] = HRoe + velRoe[:,[0]]*c; R[:,-1,ydim] = velRoe[:,[-1]]
		# # [third] row
		# R[:,ydim,0] = velRoe[:,[-1]];  R[:,ydim,1] = velRoe[:,[-1]]; 
		# R[:,ydim,-1] = velRoe[:,[-1]]; R[:,ydim,ydim] = 1.
		R = self.get_right_eigenvectors(c, evals, velRoe, HRoe)

		# Form flux Jacobian matrix multiplied by dU
		FRoe = np.matmul(R, np.expand_dims(np.abs(evals)*alphas, axis=2)).squeeze(axis=2)

		FRoe = self.undo_rotate_coord_sys(smom, FRoe, n1)

		# Left flux
		FL = physics.get_conv_flux_projected(UqL_std, n1)

		# Right flux
		FR = physics.get_conv_flux_projected(UqR_std, n1)
		
		return NN*(0.5*(FL+FR) - 0.5*FRoe)


class Roe2D(Roe1D):

	def rotate_coord_sys(self, smom, Uq, n):
		vel = self.vel
		vel[:] = Uq[:,smom]

		vel[:,0] = np.sum(Uq[:,smom]*n, axis=1)
		vel[:,1] = np.sum(Uq[:,smom]*n[:,::-1]*np.array([[-1.,1.]]), axis=1)
		
		Uq[:,smom] = vel

		return Uq

	def undo_rotate_coord_sys(self, smom, Uq, n):
		vel = self.vel
		vel[:] = Uq[:,smom]

		vel[:,0] = np.sum(Uq[:,smom]*n*np.array([[1.,-1.]]), axis=1)
		vel[:,1] = np.sum(Uq[:,smom]*n[:,::-1], axis=1)

		Uq[:,smom] = vel

		return Uq

	def get_alphas(self, c, c2, dp, dvel, drho, rhoRoe):
		alphas = self.alphas 

		alphas = super().get_alphas(c, c2, dp, dvel, drho, rhoRoe)

		alphas[:,2:3] = rhoRoe*dvel[:,-1:]

		return alphas 

	def get_eigenvalues(self, velRoe, c):
		evals = self.evals 

		evals = super().get_eigenvalues(velRoe, c)

		evals[:,2:3] = velRoe[:,0:1]

		return evals 

	def get_right_eigenvectors(self, c, evals, velRoe, HRoe):
		R = self.R

		R = super().get_right_eigenvectors(c, evals, velRoe, HRoe)

		i = 2

		# first row
		R[:,0,i] = 0.
		# second row
		R[:,1,i] = 0.
		# last row
		R[:,-1,i] = velRoe[:,-1]
		# [third] row
		R[:,i,0] = velRoe[:,-1];  R[:,i,1] = velRoe[:,-1]; 
		R[:,i,-1] = velRoe[:,-1]; R[:,i,i] = 1.

		return R 


class HLLC1D(ConvNumFluxBase):
	def __init__(self, Uq=None):
		if Uq is not None:
			n = Uq.shape[0]
			ns = Uq.shape[1]
			dim = ns - 2
		else:
			n = 0; ns = 0; dim = 0

	def compute_flux(self, physics, UqL, UqR, n):

		# Indices
		srho = physics.get_state_slice("Density")
		smom = physics.get_momentum_slice()
		srhoE = physics.get_state_slice("Energy")

		NN = np.linalg.norm(n, axis=1, keepdims=True)
		n1 = n/NN

		gamma = physics.gamma

		# unpack left hand state
		rhoL = UqL[:, srho]
		uL = UqL[:, smom]/rhoL
		unL = uL * n1
		pL = physics.compute_variable("Pressure", UqL)
		cL = physics.compute_variable("SoundSpeed", UqL)
		# unpack right hand state
		rhoR = UqR[:, srho]
		uR = UqR[:, smom]/rhoR
		unR = uR * n1
		pR = physics.compute_variable("Pressure", UqR)
		cR = physics.compute_variable("SoundSpeed", UqR)	

		# calculate averages
		rho_avg = 0.5 * (rhoL + rhoR)
		c_avg = 0.5 * (cL + cR)

		# Step 1: Get pressure estimate in the star region
		pvrs = 0.5 * ((pL + pR) - (unR - unL)*rho_avg*c_avg)
		p_star = max(0., pvrs)

		pspl = p_star / pL
		pspr = p_star / pR

		# Step 2: Get SL and SR
		qL = 1.
		if pspl > 1.:
			qL = np.sqrt(1. + (gamma + 1.) / (2.*gamma) * (pspl - 1.)) 
		SL = unL - cL*qL

		qR = 1.
		if pspr > 1.:
			qR = np.sqrt(1. + (gamma + 1.) / (2.*gamma) * (pspr - 1.))
		SR = unR + cR*qR

		# Step 3: Get shear wave speed
		raa1 = 1./(rho_avg*c_avg)
		sss = 0.5*(unL+unR) + 0.5*(pL-pR)*raa1

		# flux assembly 

		# Left State
		FL = physics.get_conv_flux_projected(UqL, n1)
		# Right State
		FR = physics.get_conv_flux_projected(UqR, n1)

		Fhllc = np.zeros_like(FL)

		if SL >= 0.:
			Fhllc = FL
		elif SR <= 0.:
			Fhllc = FR
		elif (SL <= 0.) and (sss >= 0.):
			slul = SL - unL
			cl = slul/(SL - sss)
			sssul = sss - unL
			sssel = pL / (rhoL*slul)
			ssstl = sss+sssel
			c1l = rhoL*cl*sssul
			c2l = rhoL*cl*sssul*ssstl

			Fhllc[:, srho] = FL[:, srho] + SL*(UqL[:, srho]*(cl-1.))
			Fhllc[:, smom] = FL[:, smom] + SL*(UqL[:, smom]*(cl-1.)+c1l*n1)
			Fhllc[:, srhoE] = FL[:, srhoE] + SL*(UqL[:, srhoE]*(cl-1.)+c2l)

		elif (sss <= 0.) and (SR >= 0.):
			slur = SR - unR
			cr = slur/(SR - sss)
			sssur = sss - unR
			ssser = pR / (rhoR*slur)
			ssstr = sss+ssser
			c1r = rhoR*cr*sssur
			c2r = rhoR*cr*sssur*ssstr

			Fhllc[:, srho] = FR[:, srho] + SR*(UqR[:, srho]*(cr-1.))
			Fhllc[:, smom] = FR[:, smom] + SR*(UqR[:, smom]*(cr-1.)+c1r*n1)
			Fhllc[:, srhoE] = FR[:, srhoE] + SR*(UqR[:, srhoE]*(cr-1.)+c2r)
							  
		return Fhllc