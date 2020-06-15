import numpy as np
import driver

TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : 0.5,
    "nTimeStep" : 40,
    "TimeScheme" : "RK4",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeEqSeg",
    "Solver" : "DG",
}

Output = {}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "nElem_x" : 16,
    "xmin" : -1.,
    "xmax" : 1.,
    # "PeriodicBoundariesX" : ["xmin","xmax"]
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFlux" : "LaxFriedrichs",
    "ConstVelocity" : 1.,
}

InitialCondition = {
    "Function" : "Sine",
    "omega" : 2*np.pi,
    "SetAsExact" : True,
}

BoundaryConditions = {
    "Left" : {
	    "Function" : "Sine",
	    "omega" : 2*np.pi,
    	"BCType" : "FullState",
    },
    "Right" : {
    	"Function" : "None",
    	"BCType" : "Extrapolation",
    },
}


solver, EqnSet, mesh = driver.driver(TimeStepping, Numerics, Output, Mesh,
		Physics, InitialCondition, BoundaryConditions)