import numpy as np
cfl = 0.002
num_elems = 128
dx = float(1./num_elems)
dt = cfl*dx
EndTime = 0.5
num_time_steps = int(EndTime/dt)
print(num_time_steps)
TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : EndTime,
    "num_time_steps" : num_time_steps,
    "TimeScheme" : "Strang",
    "OperatorSplitting_Imp" : "Trapezoidal",

}

Numerics = {
    "InterpOrder" : 3,
    # "InterpolateIC" : True,
    "InterpBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ConvFluxSwitch" : True,
    # "SourceTreatment" : "Implicit"
}

Output = {
    # "WriteInterval" : 2,
    # "WriteInitialSolution" : True,
    "AutoProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElems_x" : num_elems,
    # "NumElems_y" : 2,
    "xmin" : -1.,
    "xmax" : 1.,
    "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFlux" : "LaxFriedrichs",
    "ConstVelocity" : 1.,
}

nu = -3.
InitialCondition = {
    "Function" : "DampingSine",
    "omega" : 2*np.pi,
    "nu" : nu ,
    # "state" : [1.0],
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    # "Left" : {
	   #  "Function" : "DampingSine",
	   #  "omega" : 2*np.pi,
	   #  "nu" : nu,
    # 	"BCType" : "StateAll",
    # },
    # "Right" : {
    # 	#"Function" : None,
    # 	"BCType" : "Extrapolate",
    # },
}

SourceTerms = {
	"source1" : {
		"Function" : "SimpleSource",
		"nu" : nu,
	},
}
