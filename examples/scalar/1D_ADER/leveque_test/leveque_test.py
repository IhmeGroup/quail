import code
cfl = 0.05
tfinal = 0.15
dt = cfl*0.01
NumTimeSteps = int(tfinal/dt)

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : tfinal,
    "NumTimeSteps" : NumTimeSteps,
    "TimeStepper" : "ADER",
}

Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "ADERDG",
    # "ApplyLimiter" : "ScalarPositivityPreserving", 
    "SourceTreatmentADER" : "Implicit",

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

xshock = 0.3

InitialCondition = {
    "Function" : "ShockBurgers",
    "uL" : 1.,
    "uR" : 0.,
    "xshock" : xshock,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    "x1" : {
        "Function" : "ShockBurgers",
        "uL" : 1.,
        "uR" : 0.,
        "xshock" : xshock,
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
