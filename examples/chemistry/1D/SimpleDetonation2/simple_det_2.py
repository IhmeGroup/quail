import numpy as np

nelem = 100 

kv = 0.1
cfl = 0.005
FinalTime = 0.1

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "CFL" : cfl,
    # "NumTimeSteps" : 0,
    "TimeStepper" : "SSPRK3",
    # "OperatorSplittingImplicit" : "Trapezoidal",

}

Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LagrangeSeg",
    "InterpolateIC" : False,
    "Solver" : "DG",
    "ApplyLimiters" : "PositivityPreservingChem",
    # "SourceTreatmentADER" : "Implicit",
#     "NodeType" : "GaussLobatto",
#     "ElementQuadrature" : "GaussLobatto",
#     "FaceQuadrature" : "GaussLobatto",
#     "ColocatedPoints" : True,
}

Output = {
    "WriteInterval" : 50,
    "WriteInitialSolution" : True,
    "AutoProcess" : True,
    "Prefix" : "data",

}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "nElem_x" : nelem,
    "xmin" : 0.,
    "xmax" : 2.,
    # "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "Chemistry",
    "ConvFlux" : "LaxFriedrichs",
    "GasConstant" : 1.,
    "SpecificHeatRatio" : 1.4,
    "HeatRelease": 1.,
}

uL = np.array([1.4, 0., 1., 0.])
# uR = np.array([0.,0.,1.,1.])

state = {
    "Function" : "SimpleDetonation2",
    "uL" : uL,
    "xshock" : 0.3,
}

InitialCondition = state
state2 = state.copy()
state2.update({"BCType":"StateAll"})
# ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    "Left" : state2,
    # "Right" : state2,
    "Right" : { 
        "BCType" : "Extrapolate"
        }
}

SourceTerms = {
    "source1" : {
        "Function" : "Heaviside",
        "Da" : 20.,
        "Tign" : 0.22,
    },
}
