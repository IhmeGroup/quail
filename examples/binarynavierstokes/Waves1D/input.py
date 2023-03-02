import numpy as np

TimeStepping = {
	"FinalTime" : 1e-2,
	"TimeStepSize" : 0.4e-5,
	"TimeStepper" : "SSPRK3",
}

Numerics = {
	"Solver" : "DG",
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeSeg",
}

nelem = 8

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"NumElemsX" : nelem,
	"xmin" : 0.,
	"xmax" : 1.,
	"PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
	"Type" : "BinaryNavierStokes",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"DiffFluxNumerical" : "SIP",
    "R0" : 287., "R1" : 400.,
    "gamma0" : 1.4, "gamma1" : 1.1,
    "mu0" : 1., "mu1" : 2.,
    "Pr" : 0.7, "Sc" : 0.7,
}

InitialCondition = {
	"Function" : "Waves1D",
}

