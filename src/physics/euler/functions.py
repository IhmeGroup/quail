from enum import Enum, auto
import numpy as np
from physics.base.data import FcnBase, BCWeakRiemann, BCWeakPrescribed, SourceBase
from scipy.optimize import fsolve, root


class FcnType(Enum):
    SmoothIsentropicFlow = auto()
    MovingShock = auto()
    IsentropicVortex = auto()


'''
State functions
'''

def smooth_isentropic_flow(physics, fcn_data):
	if physics.Dim != 1:
		raise NotImplementedError
	irho, irhou, irhoE = physics.GetStateIndices()
	x = fcn_data.x
	t = fcn_data.Time
	U = fcn_data.U
	Data = fcn_data.Data
	gam = physics.Params["SpecificHeatRatio"]

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

	# U = np.array([r, r*u, rE]).transpose() # [nq,sr]
	U[:,irho] = r
	U[:,irhou] = r*u
	U[:,irhoE] = rE

	return U


def moving_shock(physics, fcn_data):
	irho = physics.GetStateIndex("Density")
	irhou = physics.GetStateIndex("XMomentum")
	irhoE = physics.GetStateIndex("Energy")
	if physics.Dim == 2: irhov = physics.GetStateIndex("YMomentum")
	x = fcn_data.x
	t = fcn_data.Time
	U = fcn_data.U
	Data = fcn_data.Data
	gam = physics.Params["SpecificHeatRatio"]

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
			if physics.Dim == 2: U[:, irhov] = 0.
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
		if physics.Dim == 2: U[:, irhov] = 0.
		# Energy
		U[iright, irhoE] = p1/(gam-1.) + 0.5*rho1*u1*u1
		U[ileft, irhoE] = p2/(gam-1.) + 0.5*rho2*u2*u2

	return U


def isentropic_vortex(physics, fcn_data):
	x = fcn_data.x
	t = fcn_data.Time
	U = fcn_data.U
	Data = fcn_data.Data
	gam = physics.Params["SpecificHeatRatio"]
	Rg = physics.Params["GasConstant"]

	### Parameters
	# Base flow
	try: rhob = fcn_data.rho
	except AttributeError: rhob = 1.
	# x-velocity
	try: ub = fcn_data.u
	except AttributeError: ub = 1.
	# y-velocity
	try: vb = fcn_data.v
	except AttributeError: vb = 1.
	# pressure
	try: pb = fcn_data.p
	except AttributeError: pb = 1.
	# vortex strength
	try: vs = fcn_data.vs
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


'''
Source term functions
'''