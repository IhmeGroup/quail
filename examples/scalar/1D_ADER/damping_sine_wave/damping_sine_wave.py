import numpy as np

cfl = 0.1
num_elems = 16
dx = float(1./num_elems)
dt = cfl*dx
FinalTime = 0.5
NumTimeSteps = int(FinalTime/dt)
TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "NumTimeSteps" : NumTimeSteps,
    "TimeStepper" : "ADER",
}

Numerics = {
    "SolutionOrder" : 3,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "ADERDG",
    "SourceTreatmentADER" : "Implicit",
}

Output = {
    "WriteInterval" : 2,
    "WriteInitialSolution" : True,
    "AutoPostProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : num_elems,
    # "NumElemsY" : 2,
    "xmin" : -1.,
    "xmax" : 1.,
    "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFluxNumerical" : "LaxFriedrichs",
    "ConstVelocity" : 1.,
}

nu = -1000.
InitialCondition = {
    "Function" : "DampingSine",
    "omega" : 2*np.pi,
    "nu" : nu,
}

ExactSolution = InitialCondition.copy()

# BoundaryConditions = {
#     "x1" : {
# 	    "Function" : "DampingSine",
# 	    "omega" : 2*np.pi,
# 	    "nu" : nu,
#     	"BCType" : "StateAll",
#     },
#     "x2" : {
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
