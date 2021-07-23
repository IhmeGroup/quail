import numpy as np

tfinal = 6.0
timescheme = 'ADER'
timestep = 0.001

if timescheme == 'ADER':
	solver = 'ADERDG'
	order = 4
else:
	solver = 'DG'
	order = 1

# Need to fix this script
prefix = 'time_accuracy_study/ADER/p'+str(order)+'/tf' \
	+ str(int(tfinal)) + 'dt' + str(int(timestep))
	
# prefix = 'time_accuracy_study/RK4/ref'
# prefix = 'Data'
TimeStepping = {
	"InitialTime" : 0.,
	
	"FinalTime" : tfinal,
	"TimeStepSize" : timestep,
	"TimeStepper" : timescheme,
	"OperatorSplittingImplicit" : "LSODA",
}

Numerics = {
	"SolutionOrder" : order,
	"SolutionBasis" : "LagrangeSeg",
	"Solver" : solver,
	# "SourceTreatmentADER" : "StiffImplicit",
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
		"source_treatment" : "Explicit",
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
	"Prefix" : prefix,
	"WriteFinalSolution" : True,
}