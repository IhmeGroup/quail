
TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : 0.3,
    "nTimeStep" : 300,
    "TimeScheme" : "SSPRK3",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeEqSeg",
    "Solver" : "DG",
    "ApplyLimiter" : "ScalarPositivityPreserving", 

}

Output = {
    "AutoProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "nElem_x" : 100,
    "xmin" : 0.,
    "xmax" : 1.,
    # "PeriodicBoundariesX" : ["xmin","xmax"]
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFlux" : "LaxFriedrichs",
    "ConstVelocity" : 1.,
}

InitialCondition = {
    "Function" : "ShockBurgers",
    "uL" : 1.,
    "uR" : 0.,
    "xshock" : 0.3,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    "Left" : {
        "Function" : "ShockBurgers",
        "uL" : 1.,
        "uR" : 0.,
        "xshock" : 0.3,
        "BCType" : "StateAll",
    },
    "Right" : {
        "BCType" : "Extrapolate",
    },
}
