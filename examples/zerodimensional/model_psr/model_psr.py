import numpy as np

# Case A Settings
Da = 15.89
Tinit = 1.0
tfinal = 2000.

# Case B Settings
# Da = 833.0
# Tinit = 0.15
# tfinal = 330000.

# Reference Case Settings:
timescheme = 'ODEIntegrator'
ode_scheme = 'LSODA'
solver = 'DG'
order = 0

# Operator Splitting Settings:
# timescheme = "Strang"
# solver = "DG"
# order = 0

# ADERDG Settings:
# timescheme = "ADER"
# solver = "ADERDG"
# order = 5

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : tfinal,
	"TimeStepSize" : Da / 10.,
	"TimeStepper" : timescheme,
	"OperatorSplittingImplicit" : "LSODA",
	"ODEScheme" : ode_scheme,
	
}

Numerics = {
	"SolutionOrder" : order,
	"SolutionBasis" : "LagrangeSeg",
	"Solver" : solver,
	"SourceTreatmentADER" : "StiffImplicit",
	"InterpolateFluxADER" : True,
	"PredictorGuessADER" : "ODEGuess",
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
	"T_ad" : 1.15,
	"T_in" : 0.15,
	"T_a" : 1.8
}

InitialCondition = {
	"Function" : "Uniform",
	"state" : Tinit,
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

Output = {
	"AutoPostProcess" : False,
	"WriteFinalSolution" : False,
}
