import numpy as np
import copy

FinalTime = 2.0
NumTimeSteps = 1500

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "NumTimeSteps" : NumTimeSteps,
    # "CFL" : 0.2,
    "TimeStepper" : "SSPRK3",
}

Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    # "ApplyLimiter" : "PositivityPreserving",
    "NodeType" : "Equidistant",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
    "ColocatedPoints" : False,

}

Output = {
    # "WriteInterval" : 2,
    # "WriteInitialSolution" : True,
    "AutoPostProcess" : False,
    "Prefix" : "exact",
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 100,
    "xmin" : -5.,
    "xmax" : 5.,
    # "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "Euler",
    "ConvFluxNumerical" : "Roe",
    "GasConstant" : 1.,
    "SpecificHeatRatio" : 1.4,
}

# a = 0.9
uL = np.array([1.,0.,1.])
uR = np.array([0.125,0.,0.1])

# state = { 
#     "Function" : "RiemannProblem",
#     "uL" : uL,
#     "uR" : uR,
#     # "w" : 0.05,
#     "xd" : 0.5,
# }

state_exact = {
    "Function" : "ExactRiemannSolution",
    "uL" : uL,
    "uR" : uR,
    "xmin" : -5.,
    "xmax" : 5.,
    "xd" : 0.0,
}
InitialCondition = state_exact
state2 = state_exact.copy()
state2.update({"BCType":"StateAll"})

ExactSolution = state_exact

BoundaryConditions = {
    "x1" : {
        "BCType" : "SlipWall"
        },
    "x2" : { 
        "BCType" : "SlipWall"
        }
}
