import numpy as np
# Polynomial degree (order is p+1)
p = 4
# Number of elements along the x direction
nex = 10
# Number of elements along the y direction
ney = 40
# length of the horizontal domain
d = 1.0
# Bounds of the computational domain
xmin =  0.0
xmax =  d
ymin =  0.0
ymax =  4.*d
# Grid size
h = d/nex
# Estimate of stable dt
dt  = h/(p+1)*1.0*(2.0-1.0/2.0)
# Proportionality constant for the phase-field diffusion
scl_eps = 1.5
# Equivalent grid size (element size / order of approximation)
eps = (h/(p+1))

#########################
# PHYSICAL PARAMETERS
#########################
# Atwood number
At = 0.5
# Graviational acceleration
g = 9.82
# Final normalised time
tstar = 4.0
# Final physical time
tfin = tstar*np.sqrt(d/(g*At))
# Density of fluid 1
rho01 = 1.5
# Density of fluid 2
rho02 = (1.0-At)/(At+1.)*rho01
# Reynolds number
Re = 3000
# Viscosity of both fluids
mu = (rho01*d**(1.5)*g**0.5)/Re
# Surface tension coefficient
sigma = 0.0

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : tfin,
	"TimeStepSize" : 0.0002*dt,
	"TimeStepper" : "RK4",
}

Numerics = {
	"SolutionOrder" : p,
	"SolutionBasis" : "LagrangeQuad",
	"Solver" : "DG",
	"ArtificialViscosity" : True,
	"AVParameter" : 0.75 , #0.25
}

Output = {
	"Prefix" : "Data",
	"WriteInterval" : 10,
	"WriteInitialSolution" : True,
	"WriteFinalSolution" : True,
	"AutoPostProcess" : True,
}

Mesh = {
	"ElementShape" : "Quadrilateral",
	"NumElemsX" : nex,
	"NumElemsY" : ney,
	"xmin" : xmin,
	"xmax" : xmax,
	"ymin" : ymin,
	"ymax" : ymax,
	"PeriodicBoundariesX" : ["x2", "x1"],
}

Physics = {
	"Type" : "Twophase",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"DiffFluxNumerical" : "SIP",
	"gamma1" : 1.4, #4.4
	"gamma2" : 1.4,
	"mu1" : mu,
	"mu2" : mu,
	"kappa1" : 0.0,
	"kappa2" : 0.0,
	"pinf1" : 6e4, #6e3
	"pinf2" : 6e4,
	"rho01" : rho01,
	"rho02" : rho02,
	"eps" : eps,
	"scl_eps" : scl_eps,
	"g" : g,
	"sigma" : sigma,
	"CFL_LS" : 0.01
}

b =1.0/(2.0*scl_eps*eps)

InitialCondition = {
	"Function" : "RayleighTaylor",
	"d" : d,
	"thick"  : b,
	"u" : 0.,
	"v" : 0.,
	"pressure": 0.0,
	"rho1_in" : rho01,
	"rho2_in" : rho02,
}

ExactSolution = InitialCondition.copy()

SourceTerms = {
	"Source1" : { # Name of source term ("Source1") doesn't matter
		"Function" : "BubbleSource",
	}
}

d2 = {
	"Function" : "RayleighTaylor",
	"d" : d,
	"thick"  : b,
	"u" : 0.,
	"v" : 0.,
	"pressure": 0.0,
	"rho1_in" : rho01,
	"rho2_in" : rho02,
	"BCType"  : "StateAll"
}

d3 = {
	"BCType"  : "NoSlipWall",
	"Twall" : -100.
}


BoundaryConditions = {
	"y1" : d3,
	"y2" : d2,
}
