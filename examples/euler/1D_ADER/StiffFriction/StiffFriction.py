import numpy as np

cfl = 0.1
dx = 0.05
EndTime = 0.2
nTimeStep = int(EndTime/(cfl*dx))
TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : EndTime,
    "nTimeStep" : nTimeStep,
    "TimeScheme" : "ADER",
    # "OperatorSplitting_Exp" : "SSPRK3",
    # "OperatorSplitting_Imp" : "BDF1",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeEqSeg",
    "Solver" : "ADERDG",
    "SourceTreatment" : "Implicit"
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
