import numpy as np
import copy

EndTime = 2.0
num_time_steps = 2000

TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : EndTime,
    "num_time_steps" : num_time_steps,
    # "CFL" : 0.2,
    "TimeScheme" : "SSPRK3",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeSeg",
    "Solver" : "DG",
    # "ApplyLimiter" : "PositivityPreserving",
    "InterpolateIC" : False,
    "NodeType" : "Equidistant",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
    "NodesEqualQuadpts" : False,

}

Output = {
    # "WriteInterval" : 2,
    # "WriteInitialSolution" : True,
    "AutoProcess" : False,
    "Prefix" : "exact",
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElems_x" : 100,
    "xmin" : -5.,
    "xmax" : 5.,
    # "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "Euler",
    "ConvFlux" : "Roe",
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
#     "xshock" : 0.5,
# }

state_exact = {
    "Function" : "ExactRiemannSolution",
    "uL" : uL,
    "uR" : uR,
    "xmin" : -5.,
    "xmax" : 5.,
    "xshock" : 0.0,
}
InitialCondition = state_exact
state2 = state_exact.copy()
state2.update({"BCType":"StateAll"})

ExactSolution = state_exact

BoundaryConditions = {
    "Left" : {
        "BCType" : "SlipWall"
        },
    "Right" : { 
        "BCType" : "SlipWall"
        }
}
