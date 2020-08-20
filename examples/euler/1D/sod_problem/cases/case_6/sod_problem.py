import numpy as np
import copy

FinalTime = 0.25
num_time_steps = 4000

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "num_time_steps" : num_time_steps,
    # "CFL" : 0.2,
    "TimeScheme" : "RK4",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ApplyLimiter" : "PositivityPreserving",
    "NodeType" : "GaussLobatto",
    "ElementQuadrature" : "GaussLobatto",
    "FaceQuadrature" : "GaussLobatto",
    "NodesEqualQuadpts" : False,

}

Output = {
    # "WriteInterval" : 2,
    # "WriteInitialSolution" : True,
    "AutoPostProcess" : False,
    "Prefix" : "data",
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 200,
    "xmin" : 0.,
    "xmax" : 1.,
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
#     "xshock" : 0.5,
# }

state_exact = {
    "Function" : "ExactRiemannSolution",
    "uL" : uL,
    "uR" : uR,
    "xmin" : 0.,
    "xmax" : 1.,
    "xshock" : 0.5,
}
InitialCondition = state_exact
state2 = state_exact.copy()
state2.update({"BCType":"StateAll"})

# ExactSolution = state_exact

BoundaryConditions = {
    "x1" : {
        "BCType" : "SlipWall"
        },
    "x2" : { 
        "BCType" : "SlipWall"
        }
}
