import numpy as np

cfl = 0.25
nElem = 360

FinalTime = np.pi/10.

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "CFL" : cfl,
    # "NumTimeSteps" : NumTimeStepss,
    "TimeStepper" : "Simpler",
    "OperatorSplittingImplicit" : "Trapezoidal",

}

Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LagrangeSeg",
    "InterpolateIC" : False,
    "Solver" : "DG",
    "ApplyLimiter" : "PositivityPreservingChem",
    "SourceTreatmentADER" : "Implicit",
#     "NodeType" : "GaussLobatto",
#     "ElementQuadrature" : "GaussLobatto",
#     "FaceQuadrature" : "GaussLobatto",
#     "ColocatedPoints" : True,
}

Output = {
    # "WriteInterval" : 10,
    # "WriteInitialSolution" : True,
    "AutoProcess" : False,
    "Prefix" : "TestSimpler",

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
