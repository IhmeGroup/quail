import Scalar
import numpy as np
import code
from scipy.optimize import fsolve, root
from enum import IntEnum, Enum
import Errors
import General


class Roe1DFlux(Scalar.LaxFriedrichsFlux):
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


	def ComputeFlux(self, EqnSet, UL_std, UR_std, n):
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
		imom = EqnSet.GetMomentumIndices()

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



	# def ComputeFlux(self, EqnSet, UL_old, UR_old, n):
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

	# 	# Extract helper arrays
	# 	FL = self.FL
	# 	FR = self.FR 
	# 	du = self.du 
	# 	a = self.a 
	# 	aR = self.aR 
	# 	idx = self.idx 

	# 	# Dimension
	# 	dim = EqnSet.Dim
	# 	if dim == 1:
	# 		ydim = []; # yim = [2]
	# 	else:
	# 		ydim = [2]

	# 	# Indices
	# 	irho = 0
	# 	imom = EqnSet.GetMomentumIndices()

	# 	gamma = EqnSet.Params["SpecificHeatRatio"]

	# 	NN = np.linalg.norm(n, axis=1, keepdims=True)
	# 	n1 = n/NN

	# 	# Rotated coordinate system
	# 	UL = self.RotateCoordSys(imom, UL_old, n1)
	# 	UR = self.RotateCoordSys(imom, UR_old, n1)

	# 	# Velocities
	# 	velL = UL[:,imom]/UL[:,[irho]]
	# 	velR = UR[:,imom]/UR[:,[irho]]

	# 	rhoRoe, velRoe, HRoe = self.RoeAverageState(EqnSet, irho, velL, velR, UL, UR)

	# 	# Speed of sound from Roe-averaged state
	# 	c2 = (gamma-1.)*(HRoe - 0.5*np.sum(velRoe*velRoe, axis=1, keepdims=True))
	# 	c = np.sqrt(c2)

	# 	# differences
	# 	dvel, drho, dp = self.GetDifferences(EqnSet, irho, velL, velR, UL, UR)

	# 	# alphas (left eigenvectors multipled by dU)
	# 	alphas = np.zeros_like(UL)
	# 	alphas[:,[0]] = 0.5/c2*(dp - c*rhoRoe*dvel[:,[0]])
	# 	alphas[:,[1]] = drho - dp/c2 
	# 	alphas[:,ydim] = rhoRoe*dvel[:,[-1]]
	# 	alphas[:,[-1]] = 0.5/c2*(dp + c*rhoRoe*dvel[:,[0]])

	# 	# Eigenvalues
	# 	evals = np.zeros_like(UL)
	# 	evals[:,[0]] = velRoe[:,[0]] - c
	# 	evals[:,1:-1] = velRoe[:,[0]]
	# 	evals[:,[-1]] = velRoe[:,[0]] + c

	# 	# Right eigenvector matrix
	# 	R = np.zeros(UL.shape + (UL.shape[1],))
	# 	# first row
	# 	R[:,0,[0,1,-1]] = 1.; R[:,0,ydim] = 0.
	# 	# second row
	# 	R[:,1,0] = evals[:,0]; R[:,1,1] = velRoe[:,0]; R[:,1,ydim] = 0.; R[:,1,-1] = evals[:,-1]
	# 	# last row
	# 	R[:,-1,[0]] = HRoe - velRoe[:,[0]]*c; R[:,-1,[1]] = 0.5*np.sum(velRoe*velRoe, axis=1, keepdims=True)
	# 	R[:,-1,[-1]] = HRoe + velRoe[:,[0]]*c; R[:,-1,ydim] = velRoe[:,[-1]]
	# 	# [third] row
	# 	R[:,ydim,0] = velRoe[:,[-1]];  R[:,ydim,1] = velRoe[:,[-1]]; 
	# 	R[:,ydim,-1] = velRoe[:,[-1]]; R[:,ydim,ydim] = 1.

	# 	# Form flux Jacobian matrix multiplied by dU
	# 	FRoe = np.zeros_like(UL)
	# 	FRoe[:] = np.matmul(R, np.expand_dims(np.abs(evals)*alphas, axis=2)).squeeze(axis=2)

	# 	FRoe = self.UndoRotateCoordSys(imom, FRoe, n1)

	# 	# Left flux
	# 	FL[:] = EqnSet.ConvFluxProjected(UL_old, n1)

	# 	# Right flux
	# 	FR[:] = EqnSet.ConvFluxProjected(UR_old, n1)

	# 	F = NN*(0.5*(FL+FR) - 0.5*FRoe)
	# 	# code.interact(local=locals())

	# 	return F


class Euler1D(Scalar.ConstAdvScalar):
	def __init__(self,Order,basis,mesh,StateRank=3):
		Scalar.ConstAdvScalar.__init__(self,Order,basis,mesh,StateRank) 
		# self.Params = [Rg,gam]

	def SetParams(self,**kwargs):
		Params = self.Params
		# Default values
		if not Params:
			Params["GasConstant"] = 287. # specific gas constant
			Params["SpecificHeatRatio"] = 1.4
			Params["ConvFlux"] = self.ConvFluxType["Roe"]
		# Overwrite
		for key in kwargs:
			if key not in Params.keys(): raise Exception("Input error")
			if key is "ConvFlux":
				Params[key] = self.ConvFluxType[kwargs[key]]
			else:
				Params[key] = kwargs[key]

		if Params["ConvFlux"] == self.ConvFluxType["LaxFriedrichs"]:
			self.ConvFluxFcn = Scalar.LaxFriedrichsFlux()
		elif Params["ConvFlux"] == self.ConvFluxType["Roe"]:
			if self.Dim == 1:
				self.ConvFluxFcn = Roe1DFlux()
			else:
				self.ConvFluxFcn = Roe2DFlux()

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

	def GetMomentumIndices(self):
		irV = np.zeros(self.Dim, dtype=int)
		irV[0] = self.GetStateIndex("XMomentum")
		if self.Dim == 2:
			irV[1] = self.GetStateIndex("YMomentum")

		return irV

	def ConvFluxInterior(self, u, F=None):
		dim = self.Dim
		irho = 0; irhoE = dim + 1
		imom = self.GetMomentumIndices()

		eps = General.eps

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

	def ConvFluxNumerical(self, uL, uR, NData, nq, data):
		sr = self.StateRank

		try: 
			F = data.F
		except AttributeError: 
			data.F = F = np.zeros_like(uL)
		try: 
			FL = data.FL
		except AttributeError: 
			data.FL = FL = np.zeros(sr)
		try: 
			FR = data.FR
		except AttributeError: 
			data.FR = FR = np.zeros(sr)
		try: 
			du = data.du
		except AttributeError: 
			data.du = du = np.zeros(sr)
		try: 
			lam = data.lam
		except AttributeError: 
			data.lam = lam = np.zeros(sr)

		ConvFlux = self.Params["ConvFlux"]

		gam = self.Params["SpecificHeatRatio"]

		# uL[:] = np.array([[1.3938732, -1.0008760, 1.7139886]]) # rho, rho*u, rho*E (left)
		# uR[:] = np.array([[1.3935339, -1.0001833, 1.7124593]]) # rho, rho*u, rho*E (right)

		# uL[:] = np.array([[1.3938732, -1.0008760, -0.1, 1.7139886]]) # rho, rho*u, rho*E (left)
		# uR[:] = np.array([[1.3935339, -1.0001833, -0.11, 1.7124593]]) # rho, rho*u, rho*E (right)
		# NData.nvec[:,1] = 0.1

		# if ConvFlux == self.ConvFluxType.LaxFriedrichs:
		if ConvFlux == self.ConvFluxType.LaxFriedrichs or ConvFlux == self.ConvFluxType.Roe:
			# F = self.ConvFluxLaxFriedrichs(gam, uL, uR, NData.nvec, F)
			self.ConvFluxFcn.AllocHelperArrays(uL)
			F = self.ConvFluxFcn.ComputeFlux(self, uL, uR, NData.nvec)
		else:
			for iq in range(nq):
				UL = uL[iq,:]
				UR = uR[iq,:]
				n = NData.nvec[iq*(NData.nq != 1),:]

				f = F[iq,:]

				if ConvFlux == self.ConvFluxType.Roe:
					f = self.ConvFluxRoe(gam, UL, UR, n, FL, FR, du, lam, f)

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

	def BCSlipWall(self, BC, nq, NData, uI, uB):
		irV = self.GetMomentumIndices()

		try:
			n_hat = BC.Data.n_hat
		except AttributeError:
			BC.Data.n_hat = n_hat = np.zeros_like(NData.nvec)
		if n_hat.shape != NData.nvec.shape:
			n_hat = np.zeros_like(NData.nvec)
			
		n_hat[:] = NData.nvec/np.linalg.norm(NData.nvec, axis=1, keepdims=True)

		rVn = np.sum(uI[:, irV] * n_hat, axis=1, keepdims=True)
		uB[:] = uI[:]
		uB[:, irV] -= rVn * n_hat

		return uB

	def BCPressureOutflow(self, BC, nq, NData, UI, UB):
		ir = self.GetStateIndex("Density")
		irE = self.GetStateIndex("Energy")
		irV = self.GetMomentumIndices()

		if UB is None:
			UB = np.zeros_like(UI)

		try:
			n_hat = BC.Data.n_hat
		except AttributeError:
			BC.Data.n_hat = n_hat = np.zeros_like(NData.nvec)
		n_hat = NData.nvec/np.linalg.norm(NData.nvec, axis=1, keepdims=True)

		# Pressure
		pB = BC.Data.p

		gam = self.Params["SpecificHeatRatio"]
		igam = 1./gam
		gmi = gam - 1.
		igmi = 1./gmi

		# Initialize boundary state to interior state
		UB[:] = UI[:]

		# Interior velocity in normal direction
		rhoI = UI[:,ir:ir+1]
		VnI = np.sum(UI[:,irV]*n_hat, axis=1, keepdims=True)/rhoI

		if np.any(VnI < 0.):
			print("Reversed flow on outflow boundary")

		# Compute interior pressure
		rVI2 = np.sum(UI[:,irV]**2., axis=1, keepdims=True)/rhoI
		pI = gmi*(UI[:,irE:irE+1] - 0.5*rVI2)

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
		UB[:,ir] = rhoB.reshape(-1)

		# Exterior speed of sound
		cB = np.sqrt(gam*pB/rhoB)
		dVn = 2.*igmi*(cI-cB)
		UB[:,irV] = rhoB*dVn*n_hat + rhoB*UI[:,irV]/rhoI

		# Exterior energy
		rVB2 = np.sum(UB[:,irV]**2., axis=1, keepdims=True)/rhoB
		UB[:,irE] = (pB*igmi + 0.5*rVB2).reshape(-1)

		return UB

	def BoundaryState(self, BC, nq, xglob, Time, NData, uI, uB=None):
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
			uB = self.BCSlipWall(BC, nq, NData, uI, uB)
		elif bctype == self.BCType.PressureOutflow:
			uB = self.BCPressureOutflow(BC, nq, NData, uI, uB)
		else:
			raise Exception("BC type not supported")

		return uB	

	def AdditionalScalars(self, ScalarName, U, scalar, FlagNonPhysical):
		''' Extract state variables '''
		irho = self.GetStateIndex("Density")
		irhoE = self.GetStateIndex("Energy")
		imom = self.GetMomentumIndices()
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

	def FcnSmoothIsentropicFlow(self, FcnData):
		if self.Dim != 1:
			raise NotImplementedError
		irho, irhou, irhoE = self.GetStateIndices()
		x = FcnData.x
		t = FcnData.Time
		U = FcnData.U
		Data = FcnData.Data
		gam = self.Params["SpecificHeatRatio"]

		try: a = Data.a
		except AttributeError: a = 0.9

		rho0 = lambda x,a: 1. + a*np.sin(np.pi*x)
		pressure = lambda rho,gam: rho**gam
		rho = lambda x1,x2,a: 0.5*(rho0(x1,a) + rho0(x2,a))
		vel = lambda x1,x2,a: np.sqrt(3)*(rho(x1,x2,a) - rho0(x1,a))

		f1 = lambda x1,x,t,a: x + np.sqrt(3)*rho0(x1,a)*t - x1
		f2 = lambda x2,x,t,a: x - np.sqrt(3)*rho0(x2,a)*t - x2

		x_ = x.reshape(-1)
		if isinstance(t,float):
			x1 = fsolve(f1, 0.*x_, (x_,t,a))
			if np.abs(x1.any()) > 1.: raise Exception("x1 = %g out of range" % (x1))
			x2 = fsolve(f2, 0.*x_, (x_,t,a))
			if np.abs(x2.any()) > 1.: raise Exception("x2 = %g out of range" % (x2))
		else:
			y = np.zeros(len(t))
			#for i in range(len(t)):
			#	code.interact(local=locals())
			#	y[i] = x
			y = x.transpose()
			y = y.reshape(-1)
			t = t.reshape(-1)
			x1 = root(f1, 0.*y, (y,t,a)).x
			if np.abs(x1.any()) > 1.: raise Exception("x1 = %g out of range" % (x1))
			x2 = root(f2, 0.*y, (y,t,a)).x
			if np.abs(x2.any()) > 1.: raise Exception("x2 = %g out of range" % (x2))
			
		r = rho(x1,x2,a)
		u = vel(x1,x2,a)
		p = pressure(r,gam)
		rE = p/(gam-1.) + 0.5*r*u*u

		# U = np.array([r, r*u, rE]).transpose() # [nq,sr]
		U[:,irho] = r
		U[:,irhou] = r*u
		U[:,irhoE] = rE

		return U

	def FcnMovingShock(self, FcnData):
		irho = self.GetStateIndex("Density")
		irhou = self.GetStateIndex("XMomentum")
		irhoE = self.GetStateIndex("Energy")
		if self.Dim == 2: irhov = self.GetStateIndex("YMomentum")
		x = FcnData.x
		t = FcnData.Time
		U = FcnData.U
		Data = FcnData.Data
		gam = self.Params["SpecificHeatRatio"]

		try: M = Data.M
		except AttributeError: M = 2.0
		try: xshock = Data.xshock
		except AttributeError: xshock = 0.2

		if not isinstance(t,float):
			t = t.reshape(-1)
			y = np.zeros(len(t))
			for i in range(len(t)):
				y[i]=x
			x = y

			rho1 = np.full(len(t),1.)
			p1 = np.full(len(t),1.e5)
			u1 = np.full(len(t),0.)
		else:

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
				U[iright[i], i, irho] = rho1[i]
				U[ileft[i], i, irho] = rho2[i]
				# Momentum
				U[iright[i], i, irhou] = rho1[i]*u1[i]
				U[ileft[i], i, irhou] = rho2[i]*u2[i]
				if self.Dim == 2: U[:, irhov] = 0.
				# Energy
				U[iright[i], i, irhoE] = p1[i]/(gam-1.) + 0.5*rho1[i]*u1[i]*u1[i]
				U[ileft[i], i, irhoE] = p2[i]/(gam-1.) + 0.5*rho2[i]*u2[i]*u2[i]

		else:
			# Density
			U[iright, irho] = rho1
			U[ileft, irho] = rho2
			# Momentum
			U[iright, irhou] = rho1*u1
			U[ileft, irhou] = rho2*u2
			if self.Dim == 2: U[:, irhov] = 0.
			# Energy
			U[iright, irhoE] = p1/(gam-1.) + 0.5*rho1*u1*u1
			U[ileft, irhoE] = p2/(gam-1.) + 0.5*rho2*u2*u2

		return U


class Euler2D(Euler1D):
	def __init__(self,Order,Shape,mesh,StateRank=4):
		Euler1D.__init__(self,Order,Shape,mesh,StateRank) 

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

	def ConvFluxRoe(self, gam, UL, UR, n, FL, FR, du, lam, F):
		'''
		Function: ConvFluxRoe
		-------------------
		This function computes the numerical flux (dotted with the normal)
		using the Roe flux function

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

		ir, iru, irv, irE = self.GetStateIndices()

		NN = np.linalg.norm(n)
		NN1 = 1./NN
		n1 = n/NN

		if UL[ir] <= 0. or UR[ir] <= 0.:
			raise Exception("Nonphysical density")

		# Left State
		rhol1 = 1./UL[ir]

		ul    = UL[iru]*rhol1
		vl    = UL[irv]*rhol1
		u2l   = (ul*ul + vl*vl)*UL[ir]
		unl   = (ul*n1[0] + vl*n1[1])
		pl    = (UL[irE] - 0.5*u2l  )*gmi
		hl    = (UL[irE] + pl  )*rhol1

		FL[ir]     = UL[ir]*unl
		FL[iru]    = n1[0]*pl   + UL[iru]*unl
		FL[irv]    = n1[1]*pl   + UL[irv]*unl
		FL[irE]    = (pl   + UL[irE])*unl

		# Right State
		rhor1 = 1./UR[ir]

		ur    = UR[iru]*rhor1
		vr    = UR[irv]*rhor1
		u2r   = (ur*ur + vr*vr)*UR[ir]
		unr   = (ur*n1[0] + vr*n1[1])
		pr    = (UR[irE] - 0.5*u2r  )*gmi
		hr    = (UR[irE] + pr  )*rhor1

		FR[ir ]    = UR[ir]*unr
		FR[iru]    = n1[0]*pr   + UR[iru]*unr
		FR[irv]    = n1[1]*pr   + UR[irv]*unr
		FR[irE]    = (pr   + UR[irE])*unr

		# Average state
		di     = np.sqrt(UR[ir]*rhol1)
		d1     = 1.0/(1.0+di)

		ui     = (di*ur + ul)*d1
		vi     = (di*vr + vl)*d1
		hi     = (di*hr+hl)*d1

		af     = 0.5*(ui*ui   +vi*vi )
		ucp    = ui*n1[0] + vi*n1[1]
		c2     = gmi*(hi   -af   )

		if c2 <= 0:
			raise Errors.NotPhysicalError

		ci    = np.sqrt(c2)
		ci1   = 1./ci

		# du = UR-UL
		du[ir ] = UR[ir ] - UL[ir ]
		du[iru] = UR[iru] - UL[iru]
		du[irv] = UR[irv] - UL[irv]
		du[irE] = UR[irE] - UL[irE]

		# eigenvalues
		lam[0] = ucp + ci
		lam[1] = ucp - ci
		lam[2] = ucp

		# Entropy fix
		ep     = 0.
		eps    = ep*ci
		for i in range(3):
			if lam[i] < eps and lam[i] > -eps:
				eps1 = 1./eps
				lam[i] = 0.5*(eps+lam[i]*lam[i]*eps1)

		# average and half-difference of 1st and 2nd eigs
		s1    = 0.5*(np.abs(lam[0])+np.abs(lam[1]))
		s2    = 0.5*(np.abs(lam[0])-np.abs(lam[1]))

		# third eigenvalue, absolute value
		l3    = np.abs(lam[2])

		# left eigenvector product generators
		G1    = gmi*(af*du[ir] - ui*du[iru] - vi*du[irv] + du[irE])
		G2    = -ucp*du[ir]+du[iru]*n1[0]+du[irv]*n1[1]

		# required functions of G1 and G2 
		C1    = G1*(s1-l3)*ci1*ci1 + G2*s2*ci1
		C2    = G1*s2*ci1          + G2*(s1-l3)

		# flux assembly
		F[ir ]    = NN*(0.5*(FL[ir ]+FR[ir ])-0.5*(l3*du[ir ] + C1   ))
		F[iru]    = NN*(0.5*(FL[iru]+FR[iru])-0.5*(l3*du[iru] + C1*ui + C2*n1[0]))
		F[irv]    = NN*(0.5*(FL[irv]+FR[irv])-0.5*(l3*du[irv] + C1*vi + C2*n1[1]))
		F[irE]    = NN*(0.5*(FL[irE]+FR[irE])-0.5*(l3*du[irE] + C1*hi + C2*ucp  ))

		FRoe = np.zeros([1,4])
		FRoe[0,0] = (l3*du[ir ] + C1   )
		FRoe[0,1] = (l3*du[iru] + C1*ui + C2*n1[0])
		FRoe[0,2] = (l3*du[irv] + C1*vi + C2*n1[1])
		FRoe[0,3] = (l3*du[irE] + C1*hi + C2*ucp  )
		code.interact(local=locals())

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

	def FcnIsentropicVortexPropagation(self, FcnData):
		x = FcnData.x
		t = FcnData.Time
		U = FcnData.U
		Data = FcnData.Data
		gam = self.Params["SpecificHeatRatio"]
		Rg = self.Params["GasConstant"]

		### Parameters
		# Base flow
		try: rhob = FcnData.rho
		except AttributeError: rhob = 1.
		# x-velocity
		try: ub = FcnData.u
		except AttributeError: ub = 1.
		# y-velocity
		try: vb = FcnData.v
		except AttributeError: vb = 1.
		# pressure
		try: pb = FcnData.p
		except AttributeError: pb = 1.
		# vortex strength
		try: vs = FcnData.vs
		except AttributeError: vs = 5.
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

		U[:,0] = r
		U[:,1] = ru
		U[:,2] = rv
		U[:,3] = rE

		return U










