import numpy as np

TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : 40.,
    "num_time_steps" : 500,
    "TimeScheme" : "FE",
}

Numerics = {
    "InterpOrder" : 0,
    "InterpBasis" : "LagrangeQuad",
    "Solver" : "DG",
}

Mesh = {
    "File" : "bump0.msh",
}

Physics = {
    "Type" : "Euler",
    "ConvFluxNumerical" : "Roe",
    "GasConstant" : 1.,
    "SpecificHeatRatio" : 1.4,
}

Uinflow = np.array([1.0, 0.5916079783099616, 0.0, 2.675])
InitialCondition = {
    "Function" : "Uniform",
    "state" : Uinflow,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    "inflow" : {
        "BCType" : "StateAll",
        "Function" : "Uniform",
        "state" : Uinflow,
    },
    "outflow" : {
        "BCType" : "PressureOutlet",
        "p" : 1.,
    },
    "top" : {
        "BCType" : "SlipWall",
    },
    "bottom" : {
        "BCType" : "SlipWall",
    }
}

Output = {
    "Prefix" : "p0",
    "WriteInitialSolution" : True,
    "AutoPostProcess": False,
}
