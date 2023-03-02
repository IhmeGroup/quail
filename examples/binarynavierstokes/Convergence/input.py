import numpy as np

TimeStepping = {
	"FinalTime" : 1e-2,
	"TimeStepSize" : 0.05e-5,
	"TimeStepper" : "SSPRK3",
}

Numerics = {
	"Solver" : "DG",
	"SolutionOrder" : 3,
	"SolutionBasis" : "LagrangeSeg",
}

nelem = 32

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"NumElemsX" : nelem,
	"xmin" : 0.,
	"xmax" : 1.,
}

BoundaryConditions = {
       "x1" : {
               "BCType" : "StateAll",
               "Function" : "Waves1D",
       },
       "x2" : {
               "BCType" : "StateAll",
               "Function" : "Waves1D",
       }
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

ExactSolution = InitialCondition.copy()

SourceTerms = {
	"Source1" : {
		"Function" : "ManufacturedSourceBinary",
	},
}

Output = {
	"Prefix" : "Data",
	"WriteFinalSolution" : True,
	"AutoPostProcess" : True,
}

