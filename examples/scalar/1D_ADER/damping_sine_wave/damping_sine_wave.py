import numpy as np

cfl = 0.1
nElem = 16
dx = float(1./nElem)
dt = cfl*dx
EndTime = 0.5
nTimeStep = int(EndTime/dt)
TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : EndTime,
    "nTimeStep" : nTimeStep,
    "TimeScheme" : "ADER",
}

Numerics = {
    "InterpOrder" : 3,
    "InterpBasis" : "LagrangeSeg",
    "Solver" : "ADERDG",
    "SourceTreatment" : "Implicit",
}

Output = {
    "WriteInterval" : 2,
    "WriteInitialSolution" : True,
    "AutoProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "nElem_x" : nElem,
    # "nElem_y" : 2,
    "xmin" : -1.,
    "xmax" : 1.,
    "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFlux" : "LaxFriedrichs",
    "ConstVelocity" : 1.,
}

nu = -100000.
InitialCondition = {
    "Function" : "DampingSine",
    "omega" : 2*np.pi,
    "nu" : nu,
}

ExactSolution = InitialCondition.copy()

# BoundaryConditions = {
#     "Left" : {
# 	    "Function" : "DampingSine",
# 	    "omega" : 2*np.pi,
# 	    "nu" : nu,
#     	"BCType" : "StateAll",
#     },
#     "Right" : {
#     	#"Function" : None,
#     	"BCType" : "Extrapolate",
#     },
# }

SourceTerms = {
	"source1" : {
		"Function" : "SimpleSource",
		"nu" : nu,
	},
}
