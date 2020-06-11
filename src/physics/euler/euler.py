import code
from enum import IntEnum, Enum
import numpy as np
from scipy.optimize import fsolve, root

import physics.base.base as base

import physics.euler.functions as euler_fcns
from physics.euler.functions import FcnType as euler_fcn_type
from physics.euler.functions import SourceType as euler_source_type

import errors
import general


class Roe1DFlux(base.LaxFriedrichsFlux):
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

		gamma = EqnSet.Params["SpecificHeatRatio"]

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


class Roe2DFlux(Roe1DFlux):

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


class Euler1D(base.PhysicsBase):

	StateRank = 3

	def __init__(self, order, basis, mesh):
		'''
		Method: __init__
		--------------------------------------------------------------------------
		This method initializes the temperature table. The table uses a
		piecewise linear function for the constant pressure specific heat 
		coefficients. The coefficients are selected to retain the exact 
		enthalpies at the table points.
		'''
		super().__init__(order, basis, mesh)
		# Default parameters
		self.Params.update(
			GasConstant = 287., # specific gas constant
			SpecificHeatRatio = 1.4,
			ConvFlux = self.ConvFluxType["LaxFriedrichs"]
		)

		self.fcn_map.update({
			euler_fcn_type.SmoothIsentropicFlow : euler_fcns.smooth_isentropic_flow,
			euler_fcn_type.MovingShock : euler_fcns.moving_shock,
			euler_fcn_type.DensityWave : euler_fcns.density_wave,
		})
		self.source_map.update({
			euler_source_type.StiffFriction : euler_fcns.stiff_friction,
		})

	def SetParams(self,**kwargs):
		super().SetParams(**kwargs)

		if self.Params["ConvFlux"] == self.ConvFluxType["LaxFriedrichs"]:
			self.ConvFluxFcn = base.LaxFriedrichsFlux()
		elif self.Params["ConvFlux"] == self.ConvFluxType["Roe"]:
			self.ConvFluxFcn = Roe1DFlux()

	class StateVariables(Enum):
		__Order__ = 'Density XMomentum Energy' # only needed in 2.x
		# LaTeX format
		Density = "\\rho"
		XMomentum = "\\rho u"
		Energy = "\\rho E"

	class AdditionalVariables(Enum):
	    Pressure = "p"
	    Temperature = "T"
	    Entropy = "s"
	    InternalEnergy = "\\rho e"
	    TotalEnthalpy = "H"
	    SoundSpeed = "c"
	    MaxWaveSpeed = "\\lambda"

	# class VariableType(IntEnum):
	#     Density = 0
	#     XMomentum = 1
	#     Energy = 2
	#     # Additional scalars go here
	#     Pressure = 3
	#     Temperature = 4
	#     Entropy = 5

	# class VarLabelType(Enum):
	# 	# LaTeX format
	#     Density = "\\rho"
	#     XMomentum = "\\rho u"
	#     Energy = "\\rho E"
	#     # Additional scalars go here
	#     Pressure = "p"
	#     Temperature = "T"
	#     Entropy = "s"

	class BCType(IntEnum):
	    FullState = 0
	    Extrapolation = 1
	    SlipWall = 2
	    PressureOutflow = 3

	class BCTreatment(IntEnum):
		Riemann = 0
		Prescribed = 1

	class ConvFluxType(IntEnum):
	    LaxFriedrichs = 0
	    Roe = 1

	# def GetVariableIndex(self, VariableName):
	# 	# idx = self.VariableType[VariableName]
	# 	idx = StateVariables.__members__.keys().index(VariableName)
	# 	return idx

	def GetStateIndices(self):
		irho = self.GetStateIndex("Density")
		irhou = self.GetStateIndex("XMomentum")
		irhoE = self.GetStateIndex("Energy")

		return irho, irhou, irhoE

	# def GetMomentumSlice(self):
	# 	irV = np.zeros(self.Dim, dtype=int)
	# 	irV[0] = self.GetStateIndex("XMomentum")
	# 	if self.Dim == 2:
	# 		irV[1] = self.GetStateIndex("YMomentum")

	# 	return irV

	# def GetMomentumSlice(self):
	# 	imom = np.zeros(2, dtype=int)
	# 	imom[0] = self.GetStateIndex("XMomentum")
	# 	if self.Dim == 1:
	# 		imom[1] = self.GetStateIndex("XMomentum") + 1
	# 	else:
	# 		imom[1] = self.GetStateIndex("YMomentum") + 1

	# 	return imom

	def GetMomentumSlice(self):
		irhou = self.GetStateIndex("XMomentum")
		if self.Dim == 1:
			imom = slice(irhou, irhou+1)
		else:
			irhov = self.GetStateIndex("YMomentum")
			imom = slice(irhou, irhov+1)

		return imom

	def ConvFluxInterior(self, u, F=None):
		dim = self.Dim
		irho = 0; irhoE = dim + 1
		imom = self.GetMomentumSlice()

		eps = general.eps

		rho = u[:,irho:irho+1]
		rho += eps
		rhoE = u[:,irhoE:irhoE+1]
		mom = u[:,imom]

		p = self.ComputeScalars("Pressure", u)
		h = self.ComputeScalars("TotalEnthalpy", u)

		pmat = np.zeros([u.shape[0], dim, dim])
		idx = np.full([dim,dim],False)
		np.fill_diagonal(idx,True)
		pmat[:,idx] = p

		if F is None:
			F = np.empty(u.shape+(dim,))

		F[:,irho,:] = mom
		F[:,imom,:] = np.einsum('ij,ik->ijk',mom,mom)/np.expand_dims(rho, axis=2) + pmat
		F[:,irhoE,:] = mom*h

		rho -= eps

		return F

	# def ConvFluxLaxFriedrichs(self, gam, UL, UR, n, F):
	# 	'''
	# 	Function: ConvFluxLaxFriedrichs
	# 	-------------------
	# 	This function computes the numerical flux (dotted with the normal)
	# 	using the Lax-Friedrichs flux function

	# 	INPUTS:
	# 	    gam: specific heat ratio
	# 	    UL: Left state
	# 	    UR: Right state
	# 	    n: Normal vector (assumed left to right)

	# 	OUTPUTS:
	# 	    F: Numerical flux dotted with the normal, i.e. F_hat dot n
	# 	'''

	# 	nq = F.shape[0]

	# 	# Extract intermediate arrays
	# 	# data = self.DataStorage
	# 	# try: 
	# 	# 	NN = data.NN
	# 	# except AttributeError: 
	# 	# 	data.NN = NN = np.zeros([nq,1])
	# 	# try: 
	# 	# 	n1 = data.n1
	# 	# except AttributeError: 
	# 	# 	data.n1 = n1 = np.zeros_like(n)
	# 	# try: 
	# 	# 	FL = data.FL
	# 	# except AttributeError: 
	# 	# 	data.FL = FL = np.zeros_like(F)
	# 	# try: 
	# 	# 	FL = data.FL
	# 	# except AttributeError: 
	# 	# 	data.FL = FL = np.zeros_like(F)

	# 	NN = np.linalg.norm(n, axis=1, keepdims=True)
	# 	n1 = n/NN

	# 	# Left State
	# 	FL = self.ConvFluxProjected(UL, n1)

	# 	# Right State
	# 	FR = self.ConvFluxProjected(UR, n1)

	# 	du = UR-UL

	# 	# max characteristic speed
	# 	lam = self.ComputeScalars("MaxWaveSpeed", UL, None, FlagNonPhysical=True)
	# 	lamr = self.ComputeScalars("MaxWaveSpeed", UR, None, FlagNonPhysical=True)
	# 	idx = lamr > lam
	# 	lam[idx] = lamr[idx]

	# 	# flux assembly 
	# 	F = NN*(0.5*(FL+FR) - 0.5*lam*du)

	# 	return F

	def ConvFluxRoe(self, gam, UL, UR, n, FL, FR, du, lam, F):
		'''
		Function: ConvFluxRoe
		-------------------
		This function computes the numerical flux (dotted with the normal)
		using the Roe flux function

		NOTES:

		INPUTS:
		    gam: specific heat ratio
		    UL: Left state
		    UR: Right state
		    n: Normal vector (assumed left to right)

		OUTPUTS:
		    F: Numerical flux dotted with the normal, i.e. F_hat dot n
		'''
		gmi = gam - 1.
		gmi1 = 1./gmi

		ir, iru, irE = self.GetStateIndices()

		NN = np.linalg.norm(n)
		NN1 = 1./NN
		n1 = n/NN

		# if UL[ir] <= 0. or UR[ir] <= 0.:
		# 	raise Exception("Nonphysical density")

		# Left state
		rhol1   = 1./UL[ir]
		ul      = UL[iru]*rhol1
		u2l     = (ul*ul)*UL[ir]
		unl     = (ul   *n1[0])
		pl 	    = gmi*(UL[irE] - 0.5*u2l)
		if UL[ir] <= 0. or pl <= 0.:
			raise Errors.NotPhysicalError
		cl 		= np.sqrt(gam*pl * rhol1)
		hl      = (UL[irE] + pl  )*rhol1
		FL[ir]  = UL[ir]*unl
		FL[iru] = n1[0]*pl    + UL[iru]*unl
		FL[irE] = (pl   + UL[irE])*unl

		# Right state
		rhor1   = 1./UR[ir]
		ur      = UR[iru]*rhor1
		u2r     = (ur*ur)*UR[ir]
		unr     = (ur   *n1[0])
		pr 	    = gmi*(UR[irE] - 0.5*u2r)
		if UR[ir] <= 0. or pr <= 0.:
			raise Errors.NotPhysicalError
		cr      = np.sqrt(gam*pr * rhor1)
		hr      = (UR[irE] + pr   )*rhor1
		FR[ir]  = UR[ir]*unr
		FR[iru] = n1[0]*pr    + UR[iru]*unr
		FR[irE] = (pr   + UR[irE])*unr

		# Average state
		di     = np.sqrt(UR[ir]*rhol1)
		d1     = 1.0/(1.0+di)

		ui     = (di*ur + ul)*d1
		hi     = (di*hr+hl)*d1

		af     = 0.5*(ui*ui)
		ucp    = ui*n1[0]
		c2     = gmi*(hi   -af   )

		if c2 <= 0: 
			raise Errors.NotPhysicalError

		ci    = np.sqrt(c2)
		ci1   = 1./ci

		# du = UR-UL
		du[ir ] = UR[ir ] - UL[ir ]
		du[iru] = UR[iru] - UL[iru]
		du[irE] = UR[irE] - UL[irE]

		# eigenvalues
		lam = np.zeros(3)
		lam[0] = ucp    + ci
		lam[1] = ucp    - ci
		lam[2] = ucp 

		# Entropy fix
		ep     = 0.
		eps    = ep*ci
		for i in range(3):
			if lam[i] < eps and lam[i] > -eps:
				eps1 = 1./eps
				lam[i] = 0.5*(eps+lam[i]*lam[i]*eps1)

		# define el = sign(lam[i])
		# el = np.zeros(3)
		# for i in range(3):
		# 	if lam[i] < 0:
		# 		el[i] = -1.
		# 	else:
		# 		el[i] =  1.

		# average and half-difference of 1st and 2nd eigs
		s1    = 0.5*(np.abs(lam[0])+np.abs(lam[1])) # 0.5*(el[0]*lam[0]+el[1]*lam[1])
		s2    = 0.5*(np.abs(lam[0])-np.abs(lam[1]))

		# third eigenvalue, absolute value
		l3    = np.abs(lam[2]) # el[2]*lam[2]

		# left eigenvector product generators
		G1    = gmi*(af*du[ir] - ui*du[iru] + du[irE])
		G2    = -ucp*du[ir]+du[iru]*n1[0]

		# required functions of G1 and G2 
		C1    = G1*(s1-l3)*ci1*ci1 + G2*s2*ci1
		C2    = G1*s2*ci1          + G2*(s1-l3)

		# flux assembly
		F[ir ]    = NN*(0.5*(FL[ir ]+FR[ir ])-0.5*(l3*du[ir ] + C1   ))
		F[iru]    = NN*(0.5*(FL[iru]+FR[iru])-0.5*(l3*du[iru] + C1*ui + C2*n1[0]))
		F[irE]    = NN*(0.5*(FL[irE]+FR[irE])-0.5*(l3*du[irE] + C1*hi + C2*ucp  ))

		return F

	def ConvFluxNumerical(self, uL, uR, normals): #, nq, data):
		sr = self.StateRank

		# try: 
		# 	F = data.F
		# except AttributeError: 
		# 	data.F = F = np.zeros_like(uL)
		# try: 
		# 	FL = data.FL
		# except AttributeError: 
		# 	data.FL = FL = np.zeros(sr)
		# try: 
		# 	FR = data.FR
		# except AttributeError: 
		# 	data.FR = FR = np.zeros(sr)
		# try: 
		# 	du = data.du
		# except AttributeError: 
		# 	data.du = du = np.zeros(sr)
		# try: 
		# 	lam = data.lam
		# except AttributeError: 
		# 	data.lam = lam = np.zeros(sr)

		# ConvFlux = self.Params["ConvFlux"]

		# gam = self.Params["SpecificHeatRatio"]

		# uL[:] = np.array([[1.3938732, -1.0008760, 1.7139886]]) # rho, rho*u, rho*E (left)
		# uR[:] = np.array([[1.3935339, -1.0001833, 1.7124593]]) # rho, rho*u, rho*E (right)

		# uL[:] = np.array([[1.3938732, -1.0008760, -0.1, 1.7139886]]) # rho, rho*u, rho*E (left)
		# uR[:] = np.array([[1.3935339, -1.0001833, -0.11, 1.7124593]]) # rho, rho*u, rho*E (right)
		# NData.nvec[:,1] = 0.1

		# if ConvFlux == self.ConvFluxType.LaxFriedrichs:
		# if ConvFlux == self.ConvFluxType.LaxFriedrichs or ConvFlux == self.ConvFluxType.Roe:
			# F = self.ConvFluxLaxFriedrichs(gam, uL, uR, NData.nvec, F)
		self.ConvFluxFcn.AllocHelperArrays(uL)
		F = self.ConvFluxFcn.compute_flux(self, uL, uR, normals)
		# else:
		# 	for iq in range(nq):
		# 		UL = uL[iq,:]
		# 		UR = uR[iq,:]
		# 		n = NData.nvec[iq*(NData.nq != 1),:]

		# 		f = F[iq,:]

		# 		if ConvFlux == self.ConvFluxType.Roe:
		# 			f = self.ConvFluxRoe(gam, UL, UR, n, FL, FR, du, lam, f)

		return F

	# def ConvFluxNumerical(self, uL, uR, NData, nq, data):
	# 	sr = self.StateRank

	# 	try: 
	# 		F = data.F
	# 	except AttributeError: 
	# 		data.F = F = np.zeros_like(uL)
	# 	try: 
	# 		FL = data.FL
	# 	except AttributeError: 
	# 		data.FL = FL = np.zeros(sr)
	# 	try: 
	# 		FR = data.FR
	# 	except AttributeError: 
	# 		data.FR = FR = np.zeros(sr)
	# 	try: 
	# 		du = data.du
	# 	except AttributeError: 
	# 		data.du = du = np.zeros(sr)
	# 	try: 
	# 		lam = data.lam
	# 	except AttributeError: 
	# 		data.lam = lam = np.zeros(sr)

	# 	ConvFlux = self.Params["ConvFlux"]

	# 	gam = self.Params["SpecificHeatRatio"]

	# 	for iq in range(nq):
	# 		if NData.nvec.size < nq:
	# 			n = NData.nvec[0,:]
	# 		else:
	# 			n = NData.nvec[iq,:]
	# 		UL = uL[iq,:]
	# 		UR = uR[iq,:]
	# 		#n = NData.nvec[iq,:]
	# 		f = F[iq,:]

	# 		if ConvFlux == self.ConvFluxType.HLLC:
	# 			f = self.ConvFluxHLLC(gam, UL, UR, n, FL, FR, f)
	# 		elif ConvFlux == self.ConvFluxType.LaxFriedrichs:
	# 			f = self.ConvFluxLaxFriedrichs(gam, UL, UR, n, FL, FR, du, f)
	# 		elif ConvFlux == self.ConvFluxType.Roe:
	# 			f = self.ConvFluxRoe(gam, UL, UR, n, FL, FR, du, lam, f)
	# 		else:
	# 			raise Exception("Invalid flux function")

	# 	return F

	def BCSlipWall(self, BC, nq, normals, uI, uB):
		imom = self.GetMomentumSlice()

		try:
			n_hat = BC.Data.n_hat
		except AttributeError:
			BC.Data.n_hat = n_hat = np.zeros_like(normals)
		if n_hat.shape != normals.shape:
			n_hat = np.zeros_like(normals)
			
		n_hat[:] = normals/np.linalg.norm(normals, axis=1, keepdims=True)

		rVn = np.sum(uI[:, imom] * n_hat, axis=1, keepdims=True)
		uB[:] = uI[:]
		uB[:, imom] -= rVn * n_hat

		return uB

	def BCPressureOutflow(self, BC, nq, normals, UI, UB):
		irho = self.GetStateIndex("Density")
		irhoE = self.GetStateIndex("Energy")
		imom = self.GetMomentumSlice()

		if UB is None:
			UB = np.zeros_like(UI)

		try:
			n_hat = BC.Data.n_hat
		except AttributeError:
			BC.Data.n_hat = n_hat = np.zeros_like(normals)
		n_hat = normals/np.linalg.norm(normals, axis=1, keepdims=True)

		# Pressure
		pB = BC.Data.p

		gam = self.Params["SpecificHeatRatio"]
		igam = 1./gam
		gmi = gam - 1.
		igmi = 1./gmi

		# Initialize boundary state to interior state
		UB[:] = UI[:]

		# Interior velocity in normal direction
		rhoI = UI[:,irho:irho+1]
		VnI = np.sum(UI[:,imom]*n_hat, axis=1, keepdims=True)/rhoI

		if np.any(VnI < 0.):
			print("Reversed flow on outflow boundary")

		# Compute interior pressure
		rVI2 = np.sum(UI[:,imom]**2., axis=1, keepdims=True)/rhoI
		pI = gmi*(UI[:,irhoE:irhoE+1] - 0.5*rVI2)

		if np.any(pI < 0.):
			raise Errors.NotPhysicalError

		# Interior speed of sound
		cI = np.sqrt(gam*pI/rhoI)

		# Normal Mach number
		Mn = VnI/cI
		if np.any(Mn >= 1.):
			return UB

		# Boundary density from interior entropy
		rhoB = rhoI*np.power(pB/pI, igam)
		UB[:,irho] = rhoB.reshape(-1)

		# Exterior speed of sound
		cB = np.sqrt(gam*pB/rhoB)
		dVn = 2.*igmi*(cI-cB)
		UB[:,imom] = rhoB*dVn*n_hat + rhoB*UI[:,imom]/rhoI

		# Exterior energy
		rVB2 = np.sum(UB[:,imom]**2., axis=1, keepdims=True)/rhoB
		UB[:,irhoE] = (pB*igmi + 0.5*rVB2).reshape(-1)

		return UB

	def BoundaryState(self, BC, nq, xglob, Time, normals, uI, uB=None):
		if uB is not None:
			BC.U = uB

		BC.x = xglob
		BC.nq = nq
		BC.Time = Time
		bctype = BC.BCType
		if bctype == self.BCType.FullState:
			uB = self.CallFunction(BC)
		elif bctype == self.BCType.Extrapolation:
			uB[:] = uI[:]
		elif bctype == self.BCType.SlipWall:
			uB = self.BCSlipWall(BC, nq, normals, uI, uB)
		elif bctype == self.BCType.PressureOutflow:
			uB = self.BCPressureOutflow(BC, nq, normals, uI, uB)
		else:
			raise Exception("BC type not supported")

		return uB	

	def AdditionalScalars(self, ScalarName, U, scalar, FlagNonPhysical):
		''' Extract state variables '''
		irho = self.GetStateIndex("Density")
		irhoE = self.GetStateIndex("Energy")
		imom = self.GetMomentumSlice()
		rho = U[:,irho:irho+1]
		rhoE = U[:,irhoE:irhoE+1]
		mom = U[:,imom]

		''' Common scalars '''
		gamma = self.Params["SpecificHeatRatio"]
		R = self.Params["GasConstant"]
		# Pressure
		# P = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1)/rho)
		# # Temperature
		# T = P/(rho*R)

		if FlagNonPhysical:
			if np.any(rho < 0.):
				raise Errors.NotPhysicalError

		# if np.any(P < 0.) or np.any(rho < 0.):
		# 	raise Errors.NotPhysicalError
		def getP():
			scalar[:] = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho) # just use for storage
			if FlagNonPhysical:
				if np.any(scalar < 0.):
					raise Errors.NotPhysicalError
			return scalar
		def getT():
			return getP()/(rho*R)


		''' Get final scalars '''
		sname = self.AdditionalVariables[ScalarName].name
		if sname is self.AdditionalVariables["Pressure"].name:
			scalar[:] = getP()
		elif sname is self.AdditionalVariables["Temperature"].name:
			# scalar = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho)/(rho*R)
			scalar[:] = getT()
		elif sname is self.AdditionalVariables["Entropy"].name:
			# Pressure
			# P = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho)
			# Temperature
			# T = getP()/(rho*R)

			# scalar = R*(gamma/(gamma-1.)*np.log(getT()) - np.log(getP()))
			scalar[:] = np.log(getP()/rho**gamma)
		elif sname is self.AdditionalVariables["InternalEnergy"].name:
			scalar[:] = rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho
		elif sname is self.AdditionalVariables["TotalEnthalpy"].name:
			scalar[:] = (rhoE + getP())/rho
		elif sname is self.AdditionalVariables["SoundSpeed"].name:
			# Pressure
			# P = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho)
			scalar[:] = np.sqrt(gamma*getP()/rho)
		elif sname is self.AdditionalVariables["MaxWaveSpeed"].name:
			# Pressure
			# P = GetPressure()
			scalar[:] = np.linalg.norm(mom, axis=1, keepdims=True)/rho + np.sqrt(gamma*getP()/rho)
		else:
			raise NotImplementedError

		return scalar


class Euler2D(Euler1D):

	StateRank = 4

	def __init__(self, order, basis, mesh):
		Euler1D.__init__(self, order, basis, mesh) 

	def SetParams(self,**kwargs):
		super().SetParams(**kwargs)

		if self.Params["ConvFlux"] == self.ConvFluxType["Roe"]:
			self.ConvFluxFcn = Roe2DFlux()

	class StateVariables(Enum):
		__Order__ = 'Density XMomentum YMomentum Energy' # only needed in 2.x
		# LaTeX format
		Density = "\\rho"
		XMomentum = "\\rho u"
		YMomentum = "\\rho v"
		Energy = "\\rho E"

	# class VariableType(IntEnum):
	#     Density = 0
	#     XMomentum = 1
	#     YMomentum = 2
	#     Energy = 3
	#     # Additional scalars go here
	#     Pressure = 4
	#     Temperature = 5
	#     Entropy = 6

	# class VarLabelType(Enum):
	# 	# LaTeX format
	#     Density = "\\rho"
	#     XMomentum = "\\rho u"
	#     YMomentum = "\\rho v"
	#     Energy = "\\rho E"
	#     # Additional scalars go here
	#     Pressure = "p"
	#     Temperature = "T"
	#     Entropy = "s"

	class ConvFluxType(IntEnum):
	    Roe = 0
	    LaxFriedrichs = 1

	# def GetVariableIndex(self, VariableName):
	# 	idx = self.VariableType[VariableName]
	# 	return idx

	def GetStateIndices(self):
		irho = self.GetStateIndex("Density")
		irhou = self.GetStateIndex("XMomentum")
		irhov = self.GetStateIndex("YMomentum")
		irhoE = self.GetStateIndex("Energy")

		return irho, irhou, irhov, irhoE

	# def ConvFluxInterior(self, u, F=None):
	# 	ir, iru, irv, irE = self.GetStateIndices()

	# 	r = u[:,ir]
	# 	ru = u[:,iru]
	# 	rv = u[:,irv]
	# 	rE = u[:,irE]

	# 	gam = self.Params["SpecificHeatRatio"]

	# 	p = (gam - 1.)*(rE - 0.5*(ru*ru + rv*rv)/r)
	# 	h = (rE + p)/r

	# 	if F is None:
	# 		F = np.empty(u.shape+(self.Dim,))

	# 	# x
	# 	d = 0
	# 	F[:,ir,d] = ru
	# 	F[:,iru,d] = ru*ru/r + p
	# 	F[:,irv,d] = ru*rv/r 
	# 	F[:,irE,d] = ru*h

	# 	# y
	# 	d = 1
	# 	F[:,ir,d] = rv
	# 	F[:,iru,d] = rv*ru/r 
	# 	F[:,irv,d] = rv*rv/r + p
	# 	F[:,irE,d] = rv*h

	# 	return F

	# def ConvFluxBoundary(self, BC, uI, uB, NData, nq, data):
	# 	bctreatment = self.BCTreatments[BC.BCType]
	# 	if bctreatment == self.BCTreatment.Riemann:
	# 		F = self.ConvFluxNumerical(uI, uB, NData, nq, data)
	# 	else:
	# 		# Prescribe analytic flux
	# 		try:
	# 			Fa = data.Fa
	# 		except AttributeError:
	# 			data.Fa = Fa = np.zeros([nq, self.StateRank, self.Dim])
	# 		Fa = self.ConvFluxInterior(uB, Fa)
	# 		# Take dot product with n
	# 		try: 
	# 			F = data.F
	# 		except AttributeError:
	# 			data.F = F = np.zeros_like(uI)
	# 		F[:] = np.sum(Fa.transpose(1,0,2)*NData.nvec, axis=2).transpose()

	# 	return F

	# def ConvFluxRoe(self, gam, UL, UR, n, FL, FR, du, lam, F):
	# 	'''
	# 	Function: ConvFluxRoe
	# 	-------------------
	# 	This function computes the numerical flux (dotted with the normal)
	# 	using the Roe flux function

	# 	INPUTS:
	# 	    gam: specific heat ratio
	# 	    UL: Left state
	# 	    UR: Right state
	# 	    n: Normal vector (assumed left to right)

	# 	OUTPUTS:
	# 	    F: Numerical flux dotted with the normal, i.e. F_hat dot n
	# 	'''
	# 	gmi = gam - 1.
	# 	gmi1 = 1./gmi

	# 	ir, iru, irv, irE = self.GetStateIndices()

	# 	NN = np.linalg.norm(n)
	# 	NN1 = 1./NN
	# 	n1 = n/NN

	# 	if UL[ir] <= 0. or UR[ir] <= 0.:
	# 		raise Exception("Nonphysical density")

	# 	# Left State
	# 	rhol1 = 1./UL[ir]

	# 	ul    = UL[iru]*rhol1
	# 	vl    = UL[irv]*rhol1
	# 	u2l   = (ul*ul + vl*vl)*UL[ir]
	# 	unl   = (ul*n1[0] + vl*n1[1])
	# 	pl    = (UL[irE] - 0.5*u2l  )*gmi
	# 	hl    = (UL[irE] + pl  )*rhol1

	# 	FL[ir]     = UL[ir]*unl
	# 	FL[iru]    = n1[0]*pl   + UL[iru]*unl
	# 	FL[irv]    = n1[1]*pl   + UL[irv]*unl
	# 	FL[irE]    = (pl   + UL[irE])*unl

	# 	# Right State
	# 	rhor1 = 1./UR[ir]

	# 	ur    = UR[iru]*rhor1
	# 	vr    = UR[irv]*rhor1
	# 	u2r   = (ur*ur + vr*vr)*UR[ir]
	# 	unr   = (ur*n1[0] + vr*n1[1])
	# 	pr    = (UR[irE] - 0.5*u2r  )*gmi
	# 	hr    = (UR[irE] + pr  )*rhor1

	# 	FR[ir ]    = UR[ir]*unr
	# 	FR[iru]    = n1[0]*pr   + UR[iru]*unr
	# 	FR[irv]    = n1[1]*pr   + UR[irv]*unr
	# 	FR[irE]    = (pr   + UR[irE])*unr

	# 	# Average state
	# 	di     = np.sqrt(UR[ir]*rhol1)
	# 	d1     = 1.0/(1.0+di)

	# 	ui     = (di*ur + ul)*d1
	# 	vi     = (di*vr + vl)*d1
	# 	hi     = (di*hr+hl)*d1

	# 	af     = 0.5*(ui*ui   +vi*vi )
	# 	ucp    = ui*n1[0] + vi*n1[1]
	# 	c2     = gmi*(hi   -af   )

	# 	if c2 <= 0:
	# 		raise Errors.NotPhysicalError

	# 	ci    = np.sqrt(c2)
	# 	ci1   = 1./ci

	# 	# du = UR-UL
	# 	du[ir ] = UR[ir ] - UL[ir ]
	# 	du[iru] = UR[iru] - UL[iru]
	# 	du[irv] = UR[irv] - UL[irv]
	# 	du[irE] = UR[irE] - UL[irE]

	# 	# eigenvalues
	# 	lam[0] = ucp + ci
	# 	lam[1] = ucp - ci
	# 	lam[2] = ucp

	# 	# Entropy fix
	# 	ep     = 0.
	# 	eps    = ep*ci
	# 	for i in range(3):
	# 		if lam[i] < eps and lam[i] > -eps:
	# 			eps1 = 1./eps
	# 			lam[i] = 0.5*(eps+lam[i]*lam[i]*eps1)

	# 	# average and half-difference of 1st and 2nd eigs
	# 	s1    = 0.5*(np.abs(lam[0])+np.abs(lam[1]))
	# 	s2    = 0.5*(np.abs(lam[0])-np.abs(lam[1]))

	# 	# third eigenvalue, absolute value
	# 	l3    = np.abs(lam[2])

	# 	# left eigenvector product generators
	# 	G1    = gmi*(af*du[ir] - ui*du[iru] - vi*du[irv] + du[irE])
	# 	G2    = -ucp*du[ir]+du[iru]*n1[0]+du[irv]*n1[1]

	# 	# required functions of G1 and G2 
	# 	C1    = G1*(s1-l3)*ci1*ci1 + G2*s2*ci1
	# 	C2    = G1*s2*ci1          + G2*(s1-l3)

	# 	# flux assembly
	# 	F[ir ]    = NN*(0.5*(FL[ir ]+FR[ir ])-0.5*(l3*du[ir ] + C1   ))
	# 	F[iru]    = NN*(0.5*(FL[iru]+FR[iru])-0.5*(l3*du[iru] + C1*ui + C2*n1[0]))
	# 	F[irv]    = NN*(0.5*(FL[irv]+FR[irv])-0.5*(l3*du[irv] + C1*vi + C2*n1[1]))
	# 	F[irE]    = NN*(0.5*(FL[irE]+FR[irE])-0.5*(l3*du[irE] + C1*hi + C2*ucp  ))

	# 	FRoe = np.zeros([1,4])
	# 	FRoe[0,0] = (l3*du[ir ] + C1   )
	# 	FRoe[0,1] = (l3*du[iru] + C1*ui + C2*n1[0])
	# 	FRoe[0,2] = (l3*du[irv] + C1*vi + C2*n1[1])
	# 	FRoe[0,3] = (l3*du[irE] + C1*hi + C2*ucp  )
	# 	code.interact(local=locals())

	# 	return F

	# def ConvFluxNumerical(self, uL, uR, NData, nq, data):
	# 	sr = self.StateRank

	# 	try: 
	# 		F = data.F
	# 	except AttributeError: 
	# 		data.F = F = np.zeros_like(uL)
	# 	try: 
	# 		FL = data.FL
	# 	except AttributeError: 
	# 		data.FL = FL = np.zeros(sr)
	# 	try: 
	# 		FR = data.FR
	# 	except AttributeError: 
	# 		data.FR = FR = np.zeros(sr)
	# 	try: 
	# 		du = data.du
	# 	except AttributeError: 
	# 		data.du = du = np.zeros(sr)
	# 	try: 
	# 		lam = data.lam
	# 	except AttributeError: 
	# 		data.lam = lam = np.zeros(sr)

	# 	ConvFlux = self.Params["ConvFlux"]

	# 	gam = self.Params["SpecificHeatRatio"]

	# 	if ConvFlux == self.ConvFluxType.LaxFriedrichs:
	# 		F = self.ConvFluxLaxFriedrichs(gam, uL, uR, NData.nvec, F)
	# 	else:
	# 		for iq in range(nq):
	# 			UL = uL[iq,:]
	# 			UR = uR[iq,:]
	# 			n = NData.nvec[iq*(NData.nq != 1),:]

	# 			f = F[iq,:]

	# 			if ConvFlux == self.ConvFluxType.Roe:
	# 				f = self.ConvFluxRoe(gam, UL, UR, n, FL, FR, du, lam, f)

	# 	return F

	