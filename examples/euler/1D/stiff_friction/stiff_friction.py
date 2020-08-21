import numpy as np

cfl = 0.1
dx = 0.05
FinalTime = 0.2
NumTimeSteps = int(FinalTime/(cfl*dx))
TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "NumTimeSteps" : NumTimeSteps,
    "TimeStepper" : "Strang",
    # "OperatorSplittingExplicit" : "SSPRK3",
    # "OperatorSplittingImplicit" : "BDF1",
}

Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "SourceTreatmentADER" : "Implicit"
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

nu = -100.
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
