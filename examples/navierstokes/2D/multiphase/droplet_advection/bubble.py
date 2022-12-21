import numpy as np
# Polynomial degree (order is p+1)
p = 4
# Number of elements along both directions
ne = 10
# Bounds of the computational domain
Lx = 1.0
Ly = 1.0
xmin = -Lx/2.0
xmax =  Lx/2.0
ymin = -Ly/2.0
ymax =  Ly/2.0
# Grid size
h = Lx/ne
# Estimate of stable dt
dt  = h/(p+1)*1.0*(2.0-1.0/2.0)
# Proportionality constant for the phase-field diffusion
scl_eps = 2.0
# Equivalent grid size (element size / order of approximation)
eps = (h/(p+1))

#########################
# PHYSICAL PARAMETERS
#########################

# Graviational acceleration
g = 0.0
# Density of fluid 1
rho01 = 1.
# Density of fluid 2
rho02 = 1e-3
# Viscosity of fluid 1
mu1 = 0.
# Viscosity of fluid 2
mu2 = 0.
# Surface tension coefficient
sigma = 0.01

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 1.0,
	"TimeStepSize" : 0.0003*dt,
	"TimeStepper" : "RK4",
}

Numerics = {
	"SolutionOrder" : p,
	"SolutionBasis" : "LagrangeQuad",
	"Solver" : "DG",
#	"ElementQuadrature" : "GaussLobatto",
#	"FaceQuadrature" : "GaussLobatto",
#	"NodeType" : "GaussLobatto",
#	"ColocatedPoints" : True,
#	"ApplyLimiters" : "PositivityPreservingAdvection",
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
	"NumElemsX" : ne,
	"NumElemsY" : ne,
	"xmin" : xmin,
	"xmax" : xmax,
	"ymin" : ymin,
	"ymax" : ymax,
	"PeriodicBoundariesX" : ["x2", "x1"],
	"PeriodicBoundariesY" : ["y1", "y2"],
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
	"pinf1" : 6e3, #6e3
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
	"Function" : "Bubble",
	"x0" : 0.0,
	"radius" : 25.0/89.0,
	"thick"  : b,
	"u" : 5.0,
	"v" : 0.0,
	"pressure": 1.0,
	"rho1_in" : rho01,
	"rho2_in" : rho02,
	
}

ExactSolution = InitialCondition.copy()

SourceTerms = {
	"Source1" : { # Name of source term ("Source1") doesn't matter
		"Function" : "BubbleSource",
	}
}

