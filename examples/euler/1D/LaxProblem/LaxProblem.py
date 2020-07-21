import numpy as np
import copy
cfl = 0.02
u = 1.8
dx = 0.1

dt = cfl*dx/u
EndTime = 1.3
nTimeSteps =int(EndTime/dt)
print(nTimeSteps)
NumTimeSteps = 4000
TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : EndTime,
    "NumTimeSteps" : NumTimeSteps,
    "TimeScheme" : "RK4",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ApplyLimiter" : "PositivityPreserving",
    "InterpolateIC" : True,
    "NodeType" : "Equidistant",
#    "ElementQuadrature" : "GaussLobatto",
#    "FaceQuadrature" : "GaussLobatto",
#    "NodesEqualQuadpts" : True,

}

Output = {
    # "WriteInterval" : 1,
    # "WriteInitialSolution" : True,
    "AutoProcess" : False,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "nElem_x" : 200,
    "xmin" : 0.,
    "xmax" : 1.,
    # "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "Euler",
    "ConvFlux" : "Roe",
    "GasConstant" : 1.,
    "SpecificHeatRatio" : 1.4,
}

uL = np.array([0.445, 0.698, 3.528])
uR = np.array([0.5, 0., 0.571])

state = {
    "Function" : "ExactRiemannSolution",
    "uL" : uL,
    "uR" : uR,
    "xmin" : -5.,
    "xmax" : 5.,
    "xshock" : 0.0,
}
InitialCondition = state

state2 = state.copy()
state2.update({"BCType":"StateAll"})
ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    "Left" : state2,
    "Right" : state2,
}
