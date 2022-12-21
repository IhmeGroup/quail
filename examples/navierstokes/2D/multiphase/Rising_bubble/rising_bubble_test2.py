import numpy as np
# Polynomial degree (order is p+1)
p = 4
# Number of elements along the y direction
nex = 20
# length/height ratio of the channel
ll = 2
# Number of elements along the x direction
ney = ll*nex
# Diameter of the bubble
d = 0.5
# Bounds of the computational domain
xmin =  -0.5
xmax =  0.5
ymin =  0.0
ymax =  ll
# Grid size
h = (xmax-xmin)/nex
# Estimate of stable dt
dt  = h/(p+1)*1.0*(2.0-1.0/2.0)
# Proportionality constant for the phase-field diffusion
scl_eps = 1.5
# Equivalent grid size (element size / order of approximation)
eps = (h/(p+1))
# PHYSICAL PARAMETER
# Graviational acceleration
g = 0.98
# Eotvos number
E0 = 10
# Surface tension coefficient
sigma = 1.96
# Density of fluid 1
rho01 = 1000.
# Density of fluid 2
rho02 = 0.001*rho01
# Reyndols number
Re=35
# Viscosity of fluid 1
mu1 = 10.
# Viscosity of fluid 2
mu2 = 0.01*mu1
# Intial pressure
pressure = 1.016e5

tfin = 3.
dt = 2.0e-4*eps

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : tfin,
	"TimeStepSize" : dt,
	"TimeStepper" : "RK4",
}

Numerics = {
	"SolutionOrder" : p,
	"SolutionBasis" : "LagrangeQuad",
	"Solver" : "DG",
}

snapshots = np.int(tfin/dt/100)
print(snapshots)

Output = {
	"Prefix" : "Data",
	"WriteInterval" : snapshots,
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
	"gamma1" : 4.4, #4.4
	"gamma2" : 1.4,
	"mu1" : mu1,
	"mu2" : mu2,
	"kappa1" : 0.0,
	"kappa2" : 0.0,
	"pinf1" : 1.016e6, #6e2
	"pinf2" : 0.0,
	"rho01" : rho01,
	"rho02" : rho02,
	"eps" : eps,
	"scl_eps" : scl_eps,
	"gam" : 2., #2.0
	"g" : g,
	"sigma" : sigma,
	"CFL_LS" : 0.01,
	"mdot": 0.02
}

b =1.0/(2.0*scl_eps*eps)

InitialCondition = {
	"Function" : "Rising_bubble",
	"d" : d,
	"thick"  : b,
	"pressure": pressure,
	"rho1_in" : rho01,
	"rho2_in" : rho02,
	"x0" : 0.0,
	"y0" : 0.5,
	"r0" : d/2.,
}

ExactSolution = InitialCondition.copy()

SourceTerms = {
	"Source1" : { # Name of source term ("Source1") doesn't matter
		"Function" : "BubbleSource",
	}
}


d2 = {
	"Function" : "Rising_bubble",
	"d" : d,
	"thick"  : b,
	"pressure": pressure,
	"rho1_in" : rho01,
	"rho2_in" : rho02,
	"x0" : 0.0,
	"y0" : 0.5,
	"r0" : d/2.,
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
