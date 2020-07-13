import code
cfl = 0.1
tfinal = 0.3
dt = cfl*0.01
nTimeStep = int(tfinal/dt)

TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : tfinal,
    "nTimeStep" : nTimeStep,
    "TimeScheme" : "Strang",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeSeg",
    "Solver" : "DG",
    # "ApplyLimiter" : "ScalarPositivityPreserving", 
    "SourceTreatment" : "Explicit"

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
nu = 2000.
SourceTerms = {
    "source1" : {
        "Function" : "StiffSource",
        "nu" : nu,
        "beta" : 0.5,
    },
}
