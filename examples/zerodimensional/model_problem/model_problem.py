import numpy as np

tfinal = 1.0
timescheme = 'ODEIntegrator'
timestep = 0.00048828125
solver = 'DG'
order = 0

prefix = 'Data'

TimeStepping = {
	"InitialTime" : 0.,
	
	"FinalTime" : tfinal,
	"TimeStepSize" : timestep,
	"TimeStepper" : timescheme,
	"ODEScheme" : "RK4",
}

Numerics = {
	"SolutionOrder" : order,
	"SolutionBasis" : "LagrangeSeg",
	"Solver" : solver,
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
	"Type" : "ModelProblem",
	"ConvFluxNumerical" : "LaxFriedrichs",
}

InitialCondition = {
	"Function" : "Uniform",
	"state" : np.array([1.0]),
}

SourceTerms = {
	"SimpleSource" : {
		"Function" : "SimpleSource",
		"nu" : -1.0,
		"source_treatment" : "Implicit",
	}
}

Output = {
	"AutoPostProcess" : False,
	"WriteFinalSolution" : True,
}
