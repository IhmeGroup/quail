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


class BCType(Enum):
	SlipWall = auto()
	PressureOutlet = auto()


class SourceType(Enum):
    StiffFriction = auto()


class ConvNumFluxType(Enum):
	Roe = auto()


'''
State functions
'''

class SmoothIsentropicFlow(FcnBase):
	def __init__(self, a=0.9):
		self.a = a

	def get_state(self, physics, x, t):
		
		a = self.a
		gam = physics.gamma
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


class MovingShock(FcnBase):
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
		if physics.dim == 2: irhov = physics.GetStateIndex("YMomentum")
		gam = physics.gamma
		
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
				if physics.dim == 2: Up[:, irhov] = 0.
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
			if physics.dim == 2: Up[:, irhov] = 0.
			# Energy
			Up[iright, irhoE] = p1/(gam-1.) + 0.5*rho1*u1*u1
			Up[ileft, irhoE] = p2/(gam-1.) + 0.5*rho2*u2*u2

		return Up


class DensityWave(FcnBase):
	def __init__(self, p = 1.0):
		self.p = p

	def get_state(self, physics, x, t):
		p = self.p
		irho, irhou, irhoE = physics.GetStateIndices()
		gam = physics.gamma

		Up = np.zeros([x.shape[0], physics.StateRank])

		x_ = x.reshape(-1)
		
		r = 1.0+0.1*np.sin(2.*np.pi*x_)
		ru = r*1.0
		rE = (p/(gam-1.))+0.5*ru**2/r

		Up[:,irho] = r
		Up[:,irhou] = ru
		Up[:,irhoE] = rE

		return Up


class IsentropicVortex(FcnBase):
	def __init__(self,rhob=1.,ub=1.,vb=1.,pb=1.,vs=5.):
		self.rhob = 1.
		self.ub = 1.
		self.vb = 1.
		self.pb = 1.
		self.vs = 5.
	def get_state(self,physics,x,t):		
		Up = np.zeros([x.shape[0], physics.StateRank])
		gam = physics.gamma
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
Boundary conditions
'''

class SlipWall(BCWeakPrescribed):
	def get_boundary_state(self, physics, x, t, normals, UpI):
		imom = physics.GetMomentumSlice()

		n_hat = normals/np.linalg.norm(normals, axis=1, keepdims=True)

		rhoveln = np.sum(UpI[:, imom] * n_hat, axis=1, keepdims=True)
		UpB = UpI.copy()
		UpB[:, imom] -= rhoveln * n_hat

		return UpB


class PressureOutlet(BCWeakPrescribed):
	def __init__(self, p):
		self.p = p

	def get_boundary_state(self, physics, x, t, normals, UpI):
		irho = physics.GetStateIndex("Density")
		irhoE = physics.GetStateIndex("Energy")
		imom = physics.GetMomentumSlice()

		UpB = UpI.copy()

		n_hat = normals/np.linalg.norm(normals, axis=1, keepdims=True)

		# Pressure
		pB = self.p

		gamma = physics.gamma

		# gam = physics.gamma
		# igam = 1./gam
		# gmi = gam - 1.
		# igmi = 1./gmi

		# Interior velocity in normal direction
		rhoI = UpI[:,irho:irho+1]
		velI = UpI[:,imom]/rhoI
		velnI = np.sum(velI*n_hat, axis=1, keepdims=True)

		if np.any(velnI < 0.):
			print("Incoming flow at outlet")

		# Compute interior pressure
		# rVI2 = np.sum(UpI[:,imom]**2., axis=1, keepdims=True)/rhoI
		# pI = gmi*(UpI[:,irhoE:irhoE+1] - 0.5*rVI2)
		pI = physics.ComputeScalars("Pressure", UpI)

		if np.any(pI < 0.):
			raise Errors.NotPhysicalError

		# Interior speed of sound
		# cI = np.sqrt(gam*pI/rhoI)
		cI = physics.ComputeScalars("SoundSpeed", UpI)
		JI = velnI + 2.*cI/(gamma - 1.)
		veltI = velI - velnI*n_hat

		# Normal Mach number
		Mn = velnI/cI
		if np.any(Mn >= 1.):
			return UpB

		# Boundary density from interior entropy
		rhoB = rhoI*np.power(pB/pI, 1./gamma)
		UpB[:,irho] = rhoB.reshape(-1)

		# Exterior speed of sound
		cB = np.sqrt(gamma*pB/rhoB)
		velB = (JI - 2.*cB/(gamma-1.))*n_hat + veltI
		UpB[:,imom] = rhoB*velB
		# dVn = 2.*igmi*(cI-cB)
		# UpB[:,imom] = rhoB*dVn*n_hat + rhoB*UpI[:,imom]/rhoI

		# Exterior energy
		# rVB2 = np.sum(UpB[:,imom]**2., axis=1, keepdims=True)/rhoB
		rhovel2B = rhoB*np.sum(velB**2., axis=1, keepdims=True)
		UpB[:,irhoE] = (pB/(gamma - 1.) + 0.5*rhovel2B).reshape(-1)

		return UpB


'''
Source term functions
'''

class StiffFriction(SourceBase):
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


'''
Numerical flux functions
'''

class Roe1D(ConvNumFluxBase):
	def __init__(self, u=None):
		if u is not None:
			n = u.shape[0]
			sr = u.shape[1]
			dim = sr - 2
		else:
			n = 0; sr = 0; dim = 0

		self.velL = np.zeros([n,dim])
		self.velR = np.zeros([n,dim])
		self.UL = np.zeros_like(u)
		self.UR = np.zeros_like(u)
		self.vel = np.zeros([n,dim])
		self.rhoL_sqrt = np.zeros([n,1])
		self.rhoR_sqrt = np.zeros([n,1])
		self.HL = np.zeros([n,1])
		self.HR = np.zeros([n,1])
		self.rhoRoe = np.zeros([n,1])
		self.velRoe = np.zeros([n,dim])
		self.HRoe = np.zeros([n,1])
		self.c2 = np.zeros([n,1])
		self.c = np.zeros([n,1])
		self.dvel = np.zeros([n,dim])
		self.drho = np.zeros([n,1])
		self.dp = np.zeros([n,1])
		self.alphas = np.zeros_like(u)
		self.evals = np.zeros_like(u)
		self.R = np.zeros([n,sr,sr])
		self.FRoe = np.zeros_like(u)
		self.FL = np.zeros_like(u)
		self.FR = np.zeros_like(u)

	def AllocHelperArrays(self, u):
		self.__init__(u)

	def RotateCoordSys(self, imom, U, n):
		U[:,imom] *= n

		return U

	def UndoRotateCoordSys(self, imom, U, n):
		U[:,imom] /= n

		return U

	def RoeAverageState(self, EqnSet, irho, velL, velR, uL, uR):
		rhoL_sqrt = self.rhoL_sqrt
		rhoR_sqrt = self.rhoR_sqrt
		HL = self.HL 
		HR = self.HR 

		rhoL_sqrt[:] = np.sqrt(uL[:,[irho]])
		rhoR_sqrt[:] = np.sqrt(uR[:,[irho]])
		HL[:] = EqnSet.ComputeScalars("TotalEnthalpy", uL, FlagNonPhysical=True)
		HR[:] = EqnSet.ComputeScalars("TotalEnthalpy", uR, FlagNonPhysical=True)

		self.velRoe = (rhoL_sqrt*velL + rhoR_sqrt*velR)/(rhoL_sqrt+rhoR_sqrt)
		self.HRoe = (rhoL_sqrt*HL + rhoR_sqrt*HR)/(rhoL_sqrt+rhoR_sqrt)
		self.rhoRoe = rhoL_sqrt*rhoR_sqrt

		return self.rhoRoe, self.velRoe, self.HRoe

	def GetDifferences(self, EqnSet, irho, velL, velR, uL, uR):
		dvel = self.dvel
		drho = self.drho
		dp = self.dp 

		dvel[:] = velR - velL
		drho[:] = uR[:,[irho]] - uL[:,[irho]]
		dp[:] = EqnSet.ComputeScalars("Pressure", uR) - \
			EqnSet.ComputeScalars("Pressure", uL)

		return dvel, drho, dp

	def GetAlphas(self, c, c2, dp, dvel, drho, rhoRoe):
		alphas = self.alphas 

		alphas[:,[0]] = 0.5/c2*(dp - c*rhoRoe*dvel[:,[0]])
		alphas[:,[1]] = drho - dp/c2 
		alphas[:,[-1]] = 0.5/c2*(dp + c*rhoRoe*dvel[:,[0]])

		return alphas 

	def GetEigenvalues(self, velRoe, c):
		evals = self.evals 

		evals[:,[0]] = velRoe[:,[0]] - c
		evals[:,[1]] = velRoe[:,[0]]
		evals[:,[-1]] = velRoe[:,[0]] + c

		return evals 

	def GetRightEigenvectors(self, c, evals, velRoe, HRoe):
		R = self.R

		# first row
		R[:,0,[0,1,-1]] = 1.
		# second row
		R[:,1,0] = evals[:,0]; R[:,1,1] = velRoe[:,0]; R[:,1,-1] = evals[:,-1]
		# last row
		R[:,-1,[0]] = HRoe - velRoe[:,[0]]*c; R[:,-1,[1]] = 0.5*np.sum(velRoe*velRoe, axis=1, keepdims=True)
		R[:,-1,[-1]] = HRoe + velRoe[:,[0]]*c

		return R 


	def compute_flux(self, EqnSet, UL_std, UR_std, n):
		'''
		Function: ConvFluxLaxFriedrichs
		-------------------
		This function computes the numerical flux (dotted with the normal)
		using the Lax-Friedrichs flux function

		INPUTS:
		    gam: specific heat ratio
		    UL: Left state
		    UR: Right state
		    n: Normal vector (assumed left to right)

		OUTPUTS:
		    F: Numerical flux dotted with the normal, i.e. F_hat dot n
		'''

		# Extract helper arrays
		UL = self.UL 
		UR = self.UR
		velL = self.velL
		velR = self.velR 
		c2 = self.c2
		c = self.c 
		alphas = self.alphas 
		evals = self.evals 
		R = self.R 
		FRoe = self.FRoe 
		FL = self.FL 
		FR = self.FR 

		# Indices
		irho = 0
		imom = EqnSet.GetMomentumSlice()

		gamma = EqnSet.gamma

		NN = np.linalg.norm(n, axis=1, keepdims=True)
		n1 = n/NN

		# Copy values before rotating
		UL[:] = UL_std[:]
		UR[:] = UR_std[:]

		# Rotated coordinate system
		UL = self.RotateCoordSys(imom, UL, n1)
		UR = self.RotateCoordSys(imom, UR, n1)

		# Velocities
		velL[:] = UL[:,imom]/UL[:,[irho]]
		velR[:] = UR[:,imom]/UR[:,[irho]]

		rhoRoe, velRoe, HRoe = self.RoeAverageState(EqnSet, irho, velL, velR, UL, UR)

		# Speed of sound from Roe-averaged state
		c2[:] = (gamma-1.)*(HRoe - 0.5*np.sum(velRoe*velRoe, axis=1, keepdims=True))
		c[:] = np.sqrt(c2)

		# differences
		dvel, drho, dp = self.GetDifferences(EqnSet, irho, velL, velR, UL, UR)

		# alphas (left eigenvectors multipled by dU)
		# alphas[:,[0]] = 0.5/c2*(dp - c*rhoRoe*dvel[:,[0]])
		# alphas[:,[1]] = drho - dp/c2 
		# alphas[:,ydim] = rhoRoe*dvel[:,[-1]]
		# alphas[:,[-1]] = 0.5/c2*(dp + c*rhoRoe*dvel[:,[0]])
		alphas = self.GetAlphas(c, c2, dp, dvel, drho, rhoRoe)

		# Eigenvalues
		# evals[:,[0]] = velRoe[:,[0]] - c
		# evals[:,1:-1] = velRoe[:,[0]]
		# evals[:,[-1]] = velRoe[:,[0]] + c
		evals = self.GetEigenvalues(velRoe, c)

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
		R = self.GetRightEigenvectors(c, evals, velRoe, HRoe)

		# Form flux Jacobian matrix multiplied by dU
		FRoe[:] = np.matmul(R, np.expand_dims(np.abs(evals)*alphas, axis=2)).squeeze(axis=2)

		FRoe = self.UndoRotateCoordSys(imom, FRoe, n1)

		# Left flux
		FL[:] = EqnSet.ConvFluxProjected(UL_std, n1)

		# Right flux
		FR[:] = EqnSet.ConvFluxProjected(UR_std, n1)

		return NN*(0.5*(FL+FR) - 0.5*FRoe)


class Roe2D(Roe1D):

	def RotateCoordSys(self, imom, U, n):
		vel = self.vel
		vel[:] = U[:,imom]

		vel[:,0] = np.sum(U[:,imom]*n, axis=1)
		vel[:,1] = np.sum(U[:,imom]*n[:,::-1]*np.array([[-1.,1.]]), axis=1)
		
		U[:,imom] = vel[:]

		return U

	def UndoRotateCoordSys(self, imom, U, n):
		vel = self.vel
		vel[:] = U[:,imom]

		vel[:,0] = np.sum(U[:,imom]*n*np.array([[1.,-1.]]), axis=1)
		vel[:,1] = np.sum(U[:,imom]*n[:,::-1], axis=1)

		U[:,imom] = vel[:]

		return U

	def GetAlphas(self, c, c2, dp, dvel, drho, rhoRoe):
		alphas = self.alphas 

		alphas = super().GetAlphas(c, c2, dp, dvel, drho, rhoRoe)

		alphas[:,[2]] = rhoRoe*dvel[:,[-1]]

		return alphas 

	def GetEigenvalues(self, velRoe, c):
		evals = self.evals 

		evals = super().GetEigenvalues(velRoe, c)

		evals[:,[2]] = velRoe[:,[0]]

		return evals 

	def GetRightEigenvectors(self, c, evals, velRoe, HRoe):
		R = self.R

		R = super().GetRightEigenvectors(c, evals, velRoe, HRoe)

		i = [2]

		# first row
		R[:,0,i] = 0.
		# second row
		R[:,1,i] = 0.
		# last row
		R[:,-1,i] = velRoe[:,[-1]]
		# [third] row
		R[:,i,0] = velRoe[:,[-1]];  R[:,i,1] = velRoe[:,[-1]]; 
		R[:,i,-1] = velRoe[:,[-1]]; R[:,i,i] = 1.

		return R 



