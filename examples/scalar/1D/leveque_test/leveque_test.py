import code
cfl = 0.1
tfinal = 0.3
dt = cfl*0.01
NumTimeSteps = int(tfinal/dt)

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : tfinal,
    "NumTimeSteps" : NumTimeSteps,
    "TimeStepper" : "Strang",
}

Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    # "ApplyLimiter" : "ScalarPositivityPreserving", 
    "SourceTreatmentADER" : "Explicit"

}

Output = {
    "AutoPostProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 100,
    "xmin" : 0.,
    "xmax" : 1.,
    # "PeriodicBoundariesX" : ["xmin","xmax"]
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFluxNumerical" : "LaxFriedrichs",
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
    "x1" : {
        "Function" : "ShockBurgers",
        "uL" : 1.,
        "uR" : 0.,
        "xshock" : 0.3,
        "BCType" : "StateAll",
    },
    "x2" : {
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
