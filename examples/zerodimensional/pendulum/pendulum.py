import numpy as np

tfinal = 6.0
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
	"ODEScheme" : "Trapezoidal",
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
	"Type" : "Pendulum",
	"ConvFluxNumerical" : "LaxFriedrichs",

	"g" : 9.81,
	"l" : 0.6,
}

# Dahmkohler Number
# Da = 800.0

InitialCondition = {
	"Function" : "Uniform",
	"state" : np.array([.1745, 0.]),
}

# ExactSolution = {
# 	"Function" : "PendulumExact",
# }

SourceTerms = {
	"Pendulum" : {
		"Function" : "Pendulum",
	}
	# "Mixing" : { # Name of source term ("Source1") doesn't matter
	# 	"Function" : "ScalarMixing",
	# 	"Da" : Da,
	# 	"source_treatment" : "Explicit",	
	# 	# "source_treatment" : "Implicit",	
	# },
	# "Arrhenius" : {
	# 	"Function" : "ScalarArrhenius",
	# 	"source_treatment" : "Implicit",
	# },
}

Output = {
	"AutoPostProcess" : False,
	"WriteFinalSolution" : True,
}
