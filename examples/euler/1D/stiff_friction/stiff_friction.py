import numpy as np

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 0.5,
	"NumTimeSteps" : 110,
	"TimeStepper" : "SSPRK3",
	#"TimeStepper" : "Simpler",
	"TimeStepper" : "ADER",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeSeg",
 	"Solver" : "DG",
	"Solver" : "ADERDG",
	"SourceTreatmentADER" : "SylImp",
	"PredictorGuessADER" : "Zeros",
	"PredictorThreshold" : 1e-10,
}

Output = {
	"WriteInterval" : 1,
	"WriteInitialSolution" : True,
	"AutoPostProcess" : True,
}

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"NumElemsX" : 20,
	"xmin" : 0.,
	"xmax" : 1.,
	"PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
	"Type" : "Euler",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"SpecificHeatRatio" : 1.4,
}

nu = -1.

InitialCondition = {
	"Function" : "DensityWave",
	"p" : 1.,
}

SourceTerms = {
	"source1" : {
		"Function" : "StiffFriction",
		"nu" : nu,
	},
}
