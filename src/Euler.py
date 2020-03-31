from Scalar import Scalar
import numpy as np
import code
from scipy.optimize import fsolve, root
from enum import IntEnum, Enum
import Errors


class Euler1D(Scalar):
	def __init__(self,Order,basis,mesh,StateRank=3):
		Scalar.__init__(self,Order,basis,mesh,StateRank) 
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
	    HLLC = 0
	    LaxFriedrichs = 1
	    Roe = 2

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

	def ConvFluxInterior(self, u, F):
		r = u[:,0]
		ru = u[:,1]
		rE = u[:,2]

		eps = 1e-15
		gam = self.Params["SpecificHeatRatio"]

		p = (gam - 1.)*(rE - 0.5*(ru*ru)/(r+eps))
		h = (rE + p)/(r+eps)

		if F is None:
			F = np.empty(u.shape+(self.Dim,))

		F[:,0,0] = ru
		F[:,1,0] = ru*ru/(r+eps) + p
		F[:,2,0] = ru*h

		return F

	def ConvFluxHLLC(self, gam, UL, UR, n, FL, FR, F):
		gmi = gam - 1.
		gmi1 = 1./gmi

		ir, iru, irE = self.GetStateIndices()

		NN = np.linalg.norm(n)
		NN1 = 1./NN
		n1 = n/NN

		if UL[0] <= 0. or UR[0] <= 0.:
			raise Exception("Nonphysical density")

		# Left state
		rhol1 = 1./UL[ir]

		ul = UL[iru]*rhol1
		u2l = (ul*ul)*UL[ir]
		unl = ul*n1[0]
		pl = (UL[irE] - 0.5*u2l  )*gmi
		hl = (UL[irE] + pl  )*rhol1
		al = np.sqrt(gam*pl*rhol1)

		FL[ir]     = UL[ir]*unl
		FL[iru]    = n1[0]*pl   + UL[iru]*unl
		FL[irE]    = (pl   + UL[irE])*unl

		# Right State
		rhor1 = 1./UR[ir]

		ur = UR[iru]*rhor1
		u2r = (ur*ur)*UR[ir]
		unr = (ur*n1[0])
		pr = (UR[irE] - 0.5*u2r  )*gmi
		hr = (UR[irE] + pr  )*rhor1
		ar = np.sqrt(gam*pr*rhor1)

		FR[ir ]    = UR[ir]*unr
		FR[iru]    = n1[0]*pr   + UR[iru]*unr
		FR[irE]    = (pr   + UR[irE])*unr

		# Averages
		ra = 0.5*(UL[ir]+UR[ir])

		aa = 0.5*(al+ar)

		# Step 1: Pressure estimate in the star region (PVRS)
		# See the theory guide for the process below
		ppvrs = 0.5*((pl+pr)-(unr-unl)*ra*aa)

		if ppvrs > 0.0: ps = ppvrs
		else: ps = 0.0

		pspl = ps/pl
		pspr = ps/pr

		# Step 2: ssl -> R1 head, ssr -> S3
		ql = 1.0
		if pspl > 1.0: ql = np.sqrt(1.0+(gam+1.0)/(2.0*gam)*(pspl-1.0))
		ssl = unl-al*ql

		qr = 1.0
		if pspr > 1.0: qr = np.sqrt(1.0+(gam+1.0)/(2.0*gam)*(pspr-1.0))
		ssr = unr+ar*qr

		# Step 3: Shear wave speed
		raa1 = 1.0/(ra*aa)

		sss = 0.5*(unl+unr) + 0.5*(pl-pr)*raa1

		if ssl >= 0.0:
			F[ir ] = NN*FL[ir ]
			F[iru] = NN*FL[iru]
			F[irE] = NN*FL[irE]
		elif ssr <= 0.0:
			F[ir ] = NN*FR[ir ]
			F[iru] = NN*FR[iru]
			F[irE] = NN*FR[irE]
		# Flux is approximated by one of the star-state fluxes
		elif ssl <= 0.0 and sss >= 0.0:
			slul = ssl-unl

			cl = slul/(ssl-sss)

			sssul = sss-unl

			sssel = pl/(UL[ir]*slul)

			ssstl = sss+sssel

			c1l = UL[ir]*cl*sssul

			c2l = UL[ir]*cl*sssul*ssstl

			F[ir ] = NN*(FL[ir ] + ssl*(UL[ir ]*(cl-1.0)))
			F[iru] = NN*(FL[iru] + ssl*(UL[iru]*(cl-1.0)+c1l*n1[0]))
			F[irE] = NN*(FL[irE] + ssl*(UL[irE]*(cl-1.0)+c2l))
		elif sss <= 0.0 and ssr >= 0.0:
			srur = ssr-unr

			cr = srur/(ssr-sss)

			sssur = sss-unr

			ssser = pr/(UR[ir]*srur)

			ssstr = sss+ssser

			c1r = UR[ir]*cr*sssur

			c2r = UR[ir]*cr*sssur*ssstr

			F[ir ] = NN*(FR[ir ] + ssr*(UR[ir ]*(cr-1.0)))
			F[iru] = NN*(FR[iru] + ssr*(UR[iru]*(cr-1.0)+c1r*n1[0]))
			F[irE] = NN*(FR[irE] + ssr*(UR[irE]*(cr-1.0)+c2r))

		return F

	def ConvFluxLaxFriedrichs(self, gam, UL, UR, n, FL, FR, du, F):
		gmi = gam - 1.
		gmi1 = 1./gmi

		ir, iru, irE = self.GetStateIndices()

		NN = np.linalg.norm(n)
		NN1 = 1./NN
		n1 = n/NN

		#code.interact(local=locals())
		if UL[0] <= 0. or UR[0] <= 0.:
			raise Exception("Nonphysical density")

		# Left state
		rhol1   = 1./UL[ir]
		ul      = UL[iru]*rhol1
		u2l     = (ul*ul)*UL[ir]
		unl     = (ul   *n1[0])
		pl 	    = gmi*(UL[irE] - 0.5*u2l)
		if pl <= 0.:
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
		if pr <= 0.:
			raise Errors.NotPhysicalError
		cr      = np.sqrt(gam*pr * rhor1)
		hr      = (UR[irE] + pr   )*rhor1
		FR[ir]  = UR[ir]*unr
		FR[iru] = n1[0]*pr    + UR[iru]*unr
		FR[irE] = (pr   + UR[irE])*unr

		# du = UR-UL
		du[ir ] = UR[ir ] - UL[ir ]
		du[iru] = UR[iru] - UL[iru]
		du[irE] = UR[irE] - UL[irE]

		# max characteristic speed
		lam = np.sqrt(ul*ul) + cl
		if (np.sqrt(ur*ur) + cr) > lam: 
			lam = np.sqrt(ur*ur) + cr

		# flux assembly 
		F[ir ] = NN*(0.5*(FL[ir ]+FR[ir ]) - 0.5*lam*du[ir])
		F[iru] = NN*(0.5*(FL[iru]+FR[iru]) - 0.5*lam*du[iru])
		F[irE] = NN*(0.5*(FL[irE]+FR[irE]) - 0.5*lam*du[irE])

		return F

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

		for iq in range(nq):
			if NData.nvec.size < nq:
				n = NData.nvec[0,:]
			else:
				n = NData.nvec[iq,:]
			UL = uL[iq,:]
			UR = uR[iq,:]
			#n = NData.nvec[iq,:]
			f = F[iq,:]

			if ConvFlux == self.ConvFluxType.HLLC:
				f = self.ConvFluxHLLC(gam, UL, UR, n, FL, FR, f)
			elif ConvFlux == self.ConvFluxType.LaxFriedrichs:
				f = self.ConvFluxLaxFriedrichs(gam, UL, UR, n, FL, FR, du, f)
			elif ConvFlux == self.ConvFluxType.Roe:
				f = self.ConvFluxRoe(gam, UL, UR, n, FL, FR, du, lam, f)
			else:
				raise Exception("Invalid flux function")

		return F

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

	def AdditionalScalars(self, ScalarName, U, scalar):
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

		# if np.any(P < 0.) or np.any(rho < 0.):
		# 	raise Errors.NotPhysicalError

		''' Get final scalars '''
		sname = self.AdditionalVariables[ScalarName].name
		if sname is self.AdditionalVariables["Pressure"].name:
			scalar = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho)
		elif sname is self.AdditionalVariables["Temperature"].name:
			P = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho)
			scalar = T = P/(rho*R)
		elif sname is self.AdditionalVariables["Entropy"].name:
			# Pressure
			P = (gamma - 1.)*(rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho)
			# Temperature
			T = P/(rho*R)
			scalar = R*(gamma/(gamma-1.)*np.log(T) - np.log(P))
			scalar = np.log(P/rho**gamma)
		elif sname is self.AdditionalVariables["InternalEnergy"].name:
			scalar = rhoE - 0.5*np.sum(mom*mom, axis=1, keepdims=True)/rho
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
			for i in range(len(t)):
				y[i] = x
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

	def ConvFluxInterior(self, u, F):
		ir, iru, irv, irE = self.GetStateIndices()

		r = u[:,ir]
		ru = u[:,iru]
		rv = u[:,irv]
		rE = u[:,irE]

		gam = self.Params["SpecificHeatRatio"]

		p = (gam - 1.)*(rE - 0.5*(ru*ru + rv*rv)/r)
		h = (rE + p)/r

		if F is None:
			F = np.empty(u.shape+(self.Dim,))

		# x
		d = 0
		F[:,ir,d] = ru
		F[:,iru,d] = ru*ru/r + p
		F[:,irv,d] = ru*rv/r 
		F[:,irE,d] = ru*h

		# y
		d = 1
		F[:,ir,d] = rv
		F[:,iru,d] = rv*ru/r 
		F[:,irv,d] = rv*rv/r + p
		F[:,irE,d] = rv*h

		return F

	def ConvFluxBoundary(self, BC, uI, uB, NData, nq, data):
		bctreatment = self.BCTreatments[BC.BCType]
		if bctreatment == self.BCTreatment.Riemann:
			F = self.ConvFluxNumerical(uI, uB, NData, nq, data)
		else:
			# Prescribe analytic flux
			try:
				Fa = data.Fa
			except AttributeError:
				data.Fa = Fa = np.zeros([nq, self.StateRank, self.Dim])
			Fa = self.ConvFluxInterior(uB, Fa)
			# Take dot product with n
			try: 
				F = data.F
			except AttributeError:
				data.F = F = np.zeros_like(uI)
			for jr in range(self.StateRank):
				F[:,jr] = np.sum(Fa[:,jr,:]*NData.nvec, axis=1)

			# F = np.zeros([nq, self.StateRank])
			# for iq in range(nq):
			# 	nx = NData.nvec[iq,0]; ny = NData.nvec[iq,1]
			# 	r = uB[iq,0]; ru = uB[iq,1]; rv = uB[iq,2]; rE = uB[iq,3]
			# 	run = ru*nx + rv*ny
			# 	gmi = self.Params["SpecificHeatRatio"] - 1.
			# 	p = gmi*(rE - 0.5*(ru*ru + rv*rv)/ r)
			# 	h  = (rE  + p    ) / r
			# 	F[iq,0] = run
			# 	F[iq,1] = run*ru/r + p*nx
			# 	F[iq,2] = run*rv/r + p*ny
			# 	F[iq,3] = run*h




		return F

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

		return F

	def ConvFluxLaxFriedrichs(self, gam, UL, UR, n, FL, FR, du, F):
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
		if pl <= 0.:
			raise Errors.NotPhysicalError
		cl 		= np.sqrt(gam*pl * rhol1)
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
		if pr <= 0.:
			raise Errors.NotPhysicalError
		cr      = np.sqrt(gam*pr * rhor1)
		hr    = (UR[irE] + pr  )*rhor1
		FR[ir ]    = UR[ir]*unr
		FR[iru]    = n1[0]*pr   + UR[iru]*unr
		FR[irv]    = n1[1]*pr   + UR[irv]*unr
		FR[irE]    = (pr   + UR[irE])*unr

		# du = UR-UL
		du[ir ] = UR[ir ] - UL[ir ]
		du[iru] = UR[iru] - UL[iru]
		du[irv] = UR[irv] - UL[irv]
		du[irE] = UR[irE] - UL[irE]

		# max characteristic speed
		lam = np.sqrt(ul*ul + vl*vl) + cl
		lamr = np.sqrt(ur*ur + vr*vr) + cr
		if lamr > lam: 
			lam = lamr

		# flux assembly 
		F[ir ] = NN*(0.5*(FL[ir ]+FR[ir ]) - 0.5*lam*du[ir])
		F[iru] = NN*(0.5*(FL[iru]+FR[iru]) - 0.5*lam*du[iru])
		F[irv] = NN*(0.5*(FL[irv]+FR[irv]) - 0.5*lam*du[irv])
		F[irE] = NN*(0.5*(FL[irE]+FR[irE]) - 0.5*lam*du[irE])

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

		for iq in range(nq):
			UL = uL[iq,:]
			UR = uR[iq,:]
			n = NData.nvec[iq*(NData.nq != 1),:]

			f = F[iq,:]

			if ConvFlux == self.ConvFluxType.Roe:
				f = self.ConvFluxRoe(gam, UL, UR, n, FL, FR, du, lam, f)
			elif ConvFlux == self.ConvFluxType.LaxFriedrichs:
				f = self.ConvFluxLaxFriedrichs(gam, UL, UR, n, FL, FR, du, f)
			else:
				raise Exception("Invalid flux function")

		return F

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










