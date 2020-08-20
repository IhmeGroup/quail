import numpy as np

TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : 0.1,
    # "CFL" : 0.05,
    "num_time_steps" : 100,
    "TimeScheme" : "RK4",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeSeg",
    "Solver" : "DG",
}

Output = {
    "AutoPostProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 25,
    "xmin" : -1.,
    "xmax" : 1.,
    # "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "Euler",
    "ConvFluxNumerical" : "LaxFriedrichs",
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
    "x1" : {
        "BCType" : "StateAll",
	    "Function" : "SmoothIsentropicFlow",
        "a" : a,
    },
    "x2" : {
        "BCType" : "StateAll",
        "Function" : "SmoothIsentropicFlow",
        "a" : a,
    },
}
