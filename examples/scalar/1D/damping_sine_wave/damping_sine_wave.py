import numpy as np

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : 0.5,
    "NumTimeSteps" : 40,
    "TimeStepper" : "SSPRK3",
}

Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LegendreSeg",
    "Solver" : "DG",
}

Output = {
    "AutoPostProcess" : True
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 16,
    "NumElemsY" : 2,
    "xmin" : -1.,
    "xmax" : 1.,
    "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFluxNumerical" : "LaxFriedrichs",
    "ConstVelocity" : 1.,
}

nu = -3.
InitialCondition = {
    "Function" : "DampingSine",
    "omega" : 2*np.pi,
    "nu" : nu,
}

ExactSolution = InitialCondition.copy()

SourceTerms = {
	"source1" : {
		"Function" : "SimpleSource",
		"nu" : nu,
	},
}
