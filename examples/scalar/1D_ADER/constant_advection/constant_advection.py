import numpy as np

TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : 0.5,
    "num_time_steps" : 40,
    "TimeScheme" : "ADER",
}

Numerics = {
    "InterpOrder" : 3,
    "InterpBasis" : "LagrangeSeg",
    "L2InitialCondition" : False,
    "Solver" : "ADERDG",
    "ElementQuadrature" : "GaussLobatto",
    "FaceQuadrature" : "GaussLobatto",
    "NodeType" : "GaussLobatto",
    "NodesEqualQuadpts" : True,
    "InterpolateFlux" : True,
}

Output = {
    # "WriteInterval" : 1,
    # "WriteInitialSolution" : True,
    "AutoPostProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 16,
    "xmin" : -1.,
    "xmax" : 1.,
    # "PeriodicBoundariesX" : ["xmin","xmax"]
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
