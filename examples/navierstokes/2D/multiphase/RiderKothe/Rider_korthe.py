import numpy as np
# Polynomial degree (order is p+1)
p = 3
# Number of elements along the x direction
ne = 16
# Diameter of the bubble
Lx = 1.0
Ly = 1.0
xmin = -Lx/2.0
xmax =  Lx/2.0
ymin = -Ly/2.0
ymax =  Ly/2.0
# Grid size
h = (xmax-xmin)/ne
# Proportionality constant for the phase-field diffusion
scl_eps = 2.0
# Equivalent grid size (element size / order of approximation)
eps = (h/(p+1))
# PHYSICAL PARAMETER
# Graviational acceleration
g = 0.
# Surface tension coefficient
sigma = 0.
# Density of fluid 1
rho01 = 1.
# Density of fluid 2
rho02 = 1.
# Viscosity of fluid 1
mu1 = 0.
# Viscosity of fluid 2
mu2 = 0.
# Intial pressure
pressure = 1.0

dt = 7.5e-5*eps/3.125e-2
tfin = 4.0

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

Output = {
	"Prefix" : "Data",
	"WriteInterval" : snapshots,
	"WriteInitialSolution" : True,
	"WriteFinalSolution" : True,
	"AutoPostProcess" : True,
}

Mesh = {
	"ElementShape" : "Quadrilateral",
	"NumElemsX" : ne,
	"NumElemsY" : ne,
	"xmin" : xmin,
	"xmax" : xmax,
	"ymin" : ymin,
	"ymax" : ymax,
	"PeriodicBoundariesX" : ["x1", "x2"],
	"PeriodicBoundariesY" : ["y1", "y2"],
}


Physics = {
	"Type" : "Twophase",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"DiffFluxNumerical" : "SIP",
	"gamma1" : 1.4, #4.4
	"gamma2" : 1.4,
	"mu1" : mu1,
	"mu2" : mu2,
	"kappa1" : 0.0,
	"kappa2" : 0.0,
	"pinf1" : 1.0, #6e2
	"pinf2" : 1.0,
	"rho01" : rho01,
	"rho02" : rho02,
	"eps" : eps,
	"scl_eps" : scl_eps,
	"gam" : 1.0,
	"g" : g,
	"sigma" : sigma,
	"CFL_LS" : 0.01,
	"kinetics" : 1
}

b =1.0/(2.0*scl_eps*eps)

x0 = [0.0, 0.25]

InitialCondition = {
	"Function" : "Rider_Korthe",
	"thick"  : b,
	"pressure": pressure,
	"rho1_in" : rho01,
	"rho2_in" : rho02,
	"x0" : 0.0,
	"y0" : 0.25,
	"r0" : 0.15,
}

ExactSolution = InitialCondition.copy()

SourceTerms = {
	"Source1" : { # Name of source term ("Source1") doesn't matter
		"Function" : "BubbleSource",
	}
}

d = {

	"Function" : "Rider_Korthe",
	"thick"  : b,
	"pressure": pressure,
	"rho1_in" : rho01,
	"rho2_in" : rho02,
	"x0" : 0.0,
	"y0" : 0.25,
	"r0" : 0.15,
	"BCType"  : "StateAll"
}
