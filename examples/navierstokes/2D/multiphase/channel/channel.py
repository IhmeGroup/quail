import numpy as np
# Polynomial degree (order is p+1)
p = 4
# Number of elements along the y direction
ney = 10
# length/height ratio of the channel
ll = 5
# Number of elements along the x direction
nex = ll*ney
# Diameter of the bubble
d = 0.00136
# Bounds of the computational domain
ymin =  -d/2
ymax =  d/2
xmin =  0.0
xmax =  ll*d
# Grid size
h = d/ney
# Estimate of stable dt
dt  = h/(p+1)*1.0*(2.0-1.0/2.0)
# Proportionality constant for the phase-field diffusion
scl_eps = 1.5
# Equivalent grid size (element size / order of approximation)
eps = (h/(p+1))

#########################
# PHYSICAL PARAMETERS
#########################

# Graviational acceleration
g = 9.82
# Average velocity of the Pouiseille flow
uavg = 0.1
# Pass-through caracteristic time
tconv = (xmax/uavg)
# Density of fluid 1
rho01 = 1030.
# Density of fluid 2
rho02 = 27.691
# Viscosity of fluid 1
mu1 = 1.3897e-4
# Viscosity of fluid 2
mu2 = 1.1857e-5
# Surface tension coefficient
sigma = 0.01
# Intial pressure
pressure = 1.016e5

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 1.*tconv,
	"TimeStepSize" : 0.0015*dt,
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
	"WriteInterval" : 1000,
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
	"pinf1" : 6e2, #6e2
	"pinf2" : 0.0,
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
	"Function" : "Channel",
	"d" : d,
	"thick"  : b,
	"uavg" : uavg,
	"pressure": pressure,
	"rho1_in" : rho01,
	"rho2_in" : rho02,
	"x0" : d,
	"y0" : 0,
	"r0" : d/2.26666666666666/2.0,
}

ExactSolution = InitialCondition.copy()

SourceTerms = {
	"Source1" : { # Name of source term ("Source1") doesn't matter
		"Function" : "BubbleSource",
	}
}


dwall = {
	"BCType"  : "NoSlipWall",
	"Twall" : -100.
}

din = {
	"d" : d,
	"thick"  : b,
	"uavg" : uavg,
	"rho1_in" : rho01,
	"rho2_in" : rho01,
	"BCType"  : "Subsonic_Inlet"
}

dst = {
	"Function" : "Channel",
	"d" : d,
	"thick"  : b,
	"uavg" : uavg,
	"pressure": pressure,
	"rho1_in" : rho01,
	"rho2_in" : rho01,
	"x0" : d,
	"y0" : 0,
	"r0" : d/2.26666666666666/2.0,
	"BCType"  : "StateAll"
}


BoundaryConditions = {
	"x1" : din,
	"x2" : dst,
	"y1" : dwall,
	"y2" : dwall,
}
