import numpy as np

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 2000,
	"TimeStepSize" : 1.589,
	"TimeStepper" : "Strang",
	"OperatorSplittingImplicit" : "BDF1",
}

Numerics = {
	"SolutionOrder" : 0,
	"SolutionBasis" : "LagrangeSeg",
	"Solver" : "DG",
	#"ConvFluxSwitch" : False,
}

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"NumElemsX" : 1,
	"xmin" : -1.,
	"xmax" : 1.,
	"PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
	"Type" : "ModelPSRScalar",
	"ConvFluxNumerical" : "LaxFriedrichs",
}

# Dahmkohler Number
Da = 15.89
InitialCondition = {
	"Function" : "Uniform",
	"state" : 1.,
}

SourceTerms = {
	"Mixing" : { # Name of source term ("Source1") doesn't matter
		"Function" : "ScalarMixing",
		"Da" : Da,
	},
	"Arrhenius" : {
		"Function" : "ScalarArrhenius",
	},
}
