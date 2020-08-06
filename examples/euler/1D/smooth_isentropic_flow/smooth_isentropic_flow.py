import numpy as np

TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : 0.1,
    # "CFL" : 0.05,
    "NumTimeSteps" : 100,
    "TimeScheme" : "RK4",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeSeg",
    "Solver" : "DG",
}

Output = {
    "AutoProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElems_x" : 25,
    "xmin" : -1.,
    "xmax" : 1.,
    # "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "Euler",
    "ConvFlux" : "LaxFriedrichs",
    "GasConstant" : 1.,
    "SpecificHeatRatio" : 3.,
}

a = 0.9
InitialCondition = {
    "Function" : "SmoothIsentropicFlow",
    "a" : a,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    "Left" : {
        "BCType" : "StateAll",
	    "Function" : "SmoothIsentropicFlow",
        "a" : a,
    },
    "Right" : {
        "BCType" : "StateAll",
        "Function" : "SmoothIsentropicFlow",
        "a" : a,
    },
}
