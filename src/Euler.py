from Scalar import Scalar
import numpy as np
import code
from scipy.optimize import fsolve
from enum import IntEnum, Enum

# only 1D for now
# new class for 2D or just modify this class?
class Euler(Scalar):
	def __init__(self,Order,Shape,mesh,StateRank=3):
		Scalar.__init__(self,Order,Shape,mesh,StateRank) 
		# self.Params = [Rg,gam]

	def SetParams(self,**kwargs):
		Params = self.Params
		# Default values
		if not Params:
			Params["GasConstant"] = 287. # specific gas constant
			Params["SpecificHeatRatio"] = 1.4
			Params["ConvFlux"] = self.ConvFluxType["HLLC"]
		# Overwrite
		for key in kwargs:
			if key not in Params.keys(): raise Exception("Input error")
			if key is "ConvFlux":
				Params[key] = self.ConvFluxType[kwargs[key]]
			else:
				Params[key] = kwargs[key]

	class VariableType(IntEnum):
	    Density = 0
	    XMomentum = 1
	    Energy = 2
	    # Additional scalars go here

	class VarLabelType(Enum):
		# LaTeX format
	    Density = "\\rho"
	    XMomentum = "\\rho u"
	    Energy = "\\rho E"
	    # Additional scalars go here

	class ConvFluxType(IntEnum):
	    HLLC = 0
	    LaxFriedrichs = 1
	    Roe = 2

	def GetVariableIndex(self, VariableName):
		idx = self.VariableType[VariableName]
		return idx

	def GetStateIndices(self):
		ir = self.GetVariableIndex("Density")
		iru = self.GetVariableIndex("XMomentum")
		irE = self.GetVariableIndex("Energy")

		return ir, iru, irE

	def ConvFluxInterior(self, u, F):
		r = u[:,0]
		ru = u[:,1]
		rE = u[:,2]

		gam = self.Params["SpecificHeatRatio"]

		p = (gam - 1.)*(rE - 0.5*(ru*ru)/r)
		h = (rE + p)/r

		if F is None:
			F = np.empty(u.shape+(self.Dim,))

		F[:,0,0] = ru
		F[:,1,0] = ru*ru/r + p
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

		if UL[0] <= 0. or UR[0] <= 0.:
			raise Exception("Nonphysical density")

		# Left state
		rhol1   = 1./UL[ir]
		ul      = UL[iru]*rhol1
		u2l     = (ul*ul)*UL[ir]
		unl     = (ul   *n1[0])
		pl 	    = gmi*(UL[irE] - 0.5*u2l)
		cl 		= np.sqrt(gam*pl + rhol1)
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
		gmi = gam - 1.
		gmi1 = 1./gmi

		ir, iru, irE = self.GetStateIndices()

		NN = np.linalg.norm(n)
		NN1 = 1./NN
		n1 = n/NN

		if UL[0] <= 0. or UR[0] <= 0.:
			raise Exception("Nonphysical density")

		# Left state
		rhol1   = 1./UL[ir]
		ul      = UL[iru]*rhol1
		u2l     = (ul*ul)*UL[ir]
		unl     = (ul   *n1[0])
		pl 	    = gmi*(UL[irE] - 0.5*u2l)
		cl 		= np.sqrt(gam*pl + rhol1)
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
			raise Exception("Nonphysical")

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
		# ep     = 1.e-2
		# eps    = ep*ci
		# for i in range(3):
		# 	if lam[i] < eps and lam[i] > -eps:
		# 		eps1 = 1./eps
		# 		lam[i] = 0.5*(eps+lam[i]*lam[i]*eps1)

		# define el = sign(lam[i])
		el = np.zeros(3)
		for i in range(3):
			if lam[i] < 0:
				el[i] = -1.
			else:
				el[i] =  1.

		# average and half-difference of 1st and 2nd eigs
		s1    = 0.5*(el[0]*lam[0]+el[1]*lam[1])
		s2    = 0.5*(el[0]*lam[0]-el[1]*lam[1])

		# third eigenvalue, absolute value
		l3    = el[2]*lam[2]

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
			UL = uL[iq,:]
			UR = uR[iq,:]
			n = NData.nvec[iq,:]
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

	def FcnSmoothIsentropicFlow(self, FcnData):
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
		x1 = fsolve(f1, 0.*x_, (x_,t,a))
		if np.abs(x1.any()) > 1.: raise Exception("x1 = %g out of range" % (x1))
		x2 = fsolve(f2, 0.*x_, (x_,t,a))
		if np.abs(x2.any()) > 1.: raise Exception("x2 = %g out of range" % (x2))

		r = rho(x1,x2,a)
		u = vel(x1,x2,a)
		p = pressure(r,gam)

		rE = p/(gam-1.) + 0.5*r*u*u

		# U = np.array([r, r*u, rE]).transpose() # [nq,sr]
		U[:,0] = r
		U[:,1] = r*u
		U[:,2] = rE

		return U









