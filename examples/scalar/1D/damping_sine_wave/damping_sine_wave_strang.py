import numpy as np
cfl = 0.002
num_elems = 128
dx = float(1./num_elems)
dt = cfl*dx
FinalTime = 0.5
num_time_steps = int(FinalTime/dt)
print(num_time_steps)
TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "num_time_steps" : num_time_steps,
    "TimeScheme" : "Strang",
    "OperatorSplitting_Imp" : "Trapezoidal",

}

Numerics = {
    "InterpOrder" : 3,
    "InterpBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ConvFluxSwitch" : True,
    # "SourceTreatment" : "Implicit"
}

Output = {
    # "WriteInterval" : 2,
    # "WriteInitialSolution" : True,
    "AutoPostProcess" : True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : num_elems,
    # "NumElemsY" : 2,
    "xmin" : -1.,
    "xmax" : 1.,
    "PeriodicBoundariesX" : ["x1", "x2"],
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFluxNumerical" : "LaxFriedrichs",
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
    # "x1" : {
	   #  "Function" : "DampingSine",
	   #  "omega" : 2*np.pi,
	   #  "nu" : nu,
    # 	"BCType" : "StateAll",
    # },
    # "x2" : {
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
