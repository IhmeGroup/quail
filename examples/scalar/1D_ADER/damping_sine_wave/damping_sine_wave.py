import numpy as np

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : 0.5,
    "NumTimeSteps" : 80,
    "TimeStepper" : "ADER",
}

Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "ADERDG",
    "SourceTreatmentADER" : "Implicit",
    "InterpolateFluxADER" : False,
}

Output = {
    "WriteInterval" : 2,
    "WriteInitialSolution" : True,
    "AutoPostProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 16,
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

SourceTerms = {
	"source1" : { # Name of source term ("Source1") doesn't matter
		"Function" : "SimpleSource",
		"nu" : nu,
	},
}
