import numpy as np

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : 0.5,
    "CFL" : 0.1,
    # "TimeStepSize" : 0.0125,
    # "NumTimeSteps" : 40,
    "TimeStepper" : "RK4",
}

Numerics = {
    "SolutionOrder" : 3,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ElementQuadrature" : "GaussLobatto",
    "FaceQuadrature" : "GaussLobatto",
    "NodeType" : "GaussLobatto",
    "ColocatedPoints" : True,
    "InterpolateFluxADER" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 16,
    "xmin" : -1.,
    "xmax" : 1.,
    #"PeriodicBoundariesX" : ["x1","x2"]
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFluxNumerical" : "LaxFriedrichs",
    "ConstVelocity" : 1.,
}

InitialCondition = {
    "Function" : "Sine",
    "omega" : 2*np.pi,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    "x1" : {
	    "Function" : "Sine",
	    "omega" : 2*np.pi,
   	"BCType" : "StateAll",
    },
    "x2" : {
    	"BCType" : "Extrapolate",
    },
}
