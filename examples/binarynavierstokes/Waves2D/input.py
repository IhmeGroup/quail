import numpy as np

TimeStepping = {
	"FinalTime" : 1e-2,
	"TimeStepSize" : 0.2e-5,
	"TimeStepper" : "SSPRK3",
}

Numerics = {
	"Solver" : "DG",
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeQuad",
}

nelem = 8

Mesh = {
	"ElementShape" : "Quadrilateral",
	"NumElemsX" : nelem,
	"NumElemsY" : nelem,
	"xmin" : 0.,
	"xmax" : 1.,
	"ymin" : 0.,
	"ymax" : 1.,
	"PeriodicBoundariesX" : ["x2", "x1"],
	"PeriodicBoundariesY" : ["y1", "y2"],
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
	"Function" : "Waves2D",
}

