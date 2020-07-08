import numpy as np
import copy
cfl = 0.05
u = 1.
dx = 0.1

dt = cfl*dx/u
EndTime = 2.0
nTimeSteps =int(EndTime/dt)
print(nTimeSteps)
TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : EndTime,
    "nTimeStep" : nTimeSteps,
    "TimeScheme" : "ADER",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeEqSeg",
    "Solver" : "ADERDG",
    "ApplyLimiter" : "PositivityPreserving",
    "InterpolateIC" : True,
    "NodeType" : "GaussLobatto",
    "ElementQuadrature" : "GaussLobatto",
    "FaceQuadrature" : "GaussLobatto",
    "NodesEqualQuadpts" : True,

}

Output = {
    # "WriteInterval" : 5,
    "WriteInitialSolution" : True,
    "AutoProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "nElem_x" : 100,
    "xmin" : -5.,
    "xmax" : 5.,
    # "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "Euler",
    "ConvFlux" : "LaxFriedrichs",
    # "GasConstant" : 1.,
    # "SpecificHeatRatio" : 3.,
}

# a = 0.9
uL = np.array([1.,0.,1.])
uR = np.array([0.125,0.,0.1])

state = { 
    "Function" : "SmoothRiemannProblem",
    "uL" : uL,
    "uR" : uR,
    "w" : 0.05,
    "xshock" : 0.,
}
InitialCondition = state
state2 = state.copy()
state2.update({"BCType":"StateAll"})
# ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    "Left" : state2,
    "Right" : state2,
}
