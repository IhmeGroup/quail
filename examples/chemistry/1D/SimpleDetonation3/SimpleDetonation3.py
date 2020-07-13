import numpy as np

cfl = 0.5
nElem = 1000
dx = float(1/nElem)
u = 5.
dt = cfl*dx/u

EndTime = np.pi/10.
nTimeSteps =  int(EndTime/dt)
print(nTimeSteps)
TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : EndTime,
    "nTimeStep" : nTimeSteps,
    "TimeScheme" : "SSPRK3",
    # "OperatorSplitting_Imp" : "Trapezoidal",

}

Numerics = {
    "InterpOrder" : 0,
    "InterpBasis" : "LagrangeEqSeg",
    "InterpolateIC" : True,
    "Solver" : "DG",
    "ApplyLimiter" : "PositivityPreservingChem",
    "SourceTreatment" : "Implicit",
#     "NodeType" : "GaussLobatto",
#     "ElementQuadrature" : "GaussLobatto",
#     "FaceQuadrature" : "GaussLobatto",
#     "NodesEqualQuadpts" : True,
}

Output = {
    # "WriteInterval" : 10,
    # "WriteInitialSolution" : True,
    "AutoProcess" : False,
    "Prefix" : "Data",

}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "nElem_x" : nElem,
    "xmin" : 0.,
    "xmax" : 2.*np.pi,
    # "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "Chemistry",
    "ConvFlux" : "LaxFriedrichs",
    "GasConstant" : 1.,
    "SpecificHeatRatio" : 1.2,
    "HeatRelease": 50.,
}

uL = np.array([2.,4.,40.,0.])
uR = np.array([0.,0.,1.,1.])

state = {
    "Function" : "SimpleDetonation3",
    "uL" : uL,
    "uR" : uR,
    "xshock" : np.pi/2.,
}

InitialCondition = state
state2 = state.copy()
state2.update({"BCType":"StateAll"})
# ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    "Left" : state2,
    # "Right" : state2,
    "Right" : { 
        "BCType" : "SlipWall"
        }
}

SourceTerms = {
    "source1" : {
        "Function" : "Heaviside",
        "Da" : 3000.,
        "Tign" : 2.,
    },
}
