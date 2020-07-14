import numpy as np

cfl = 0.1
dx = 0.05
EndTime = 0.2
NumTimeSteps = int(EndTime/(cfl*dx))
TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : EndTime,
    "NumTimeSteps" : NumTimeSteps,
    "TimeScheme" : "Strang",
    # "OperatorSplitting_Exp" : "SSPRK3",
    # "OperatorSplitting_Imp" : "BDF1",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "SourceTreatment" : "Implicit"
}

Output = {
    "WriteInterval" : 1,
    "WriteInitialSolution" : True,
    "AutoProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "nElem_x" : 20,
    "xmin" : 0.,
    "xmax" : 1.,
    "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "Euler",
    "ConvFlux" : "LaxFriedrichs",
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
