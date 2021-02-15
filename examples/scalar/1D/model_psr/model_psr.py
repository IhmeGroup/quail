import numpy as np

TimeStepping = {
	"InitialTime" : 0.,
	# "FinalTime" : 2000.,
	"FinalTime" : 330000.,
	# "TimeStepSize" : 1.589,
	"TimeStepSize" : 83.3/16.,
	"TimeStepper" : "Strang",
	# "TimeStepper" : "ADER", 
	# "OperatorSplittingImplicit" : "Scipy",
	"OperatorSplittingImplicit" : "Trapezoidal",
	# "OperatorSplittingExplicit" : "RK4",
}

Numerics = {
	"SolutionOrder" : 1,
	"SolutionBasis" : "LagrangeSeg",
	# "Solver" : "ADERDG",
	"Solver" : "DG",
	"SourceTreatmentADER" : "Testing",
	# "ElementQuadrature" : "GaussLobatto",
	# "FaceQuadrature" : "GaussLobatto",

	"InterpolateFluxADER" : False,
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
Da = 833.0
# Da = 15.89
InitialCondition = {
	"Function" : "Uniform",
	"state" : .15,
	# "state" : 1.,
}

SourceTerms = {
	"Mixing" : { # Name of source term ("Source1") doesn't matter
		"Function" : "ScalarMixing",
		"Da" : Da,
		"source_treatment" : "Explicit",	
	},
	"Arrhenius" : {
		"Function" : "ScalarArrhenius",
		"source_treatment" : "Implicit",
	},
}
