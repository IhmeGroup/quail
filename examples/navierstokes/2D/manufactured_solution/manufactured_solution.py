TimeStepping = {
	"FinalTime" : 0.5,
	#"NumTimeSteps" : 0,
	# "CFL" : 0.001,
	"TimeStepSize" : 0.00125,
	"TimeStepper" : "SSPRK3",
}

Numerics = {
	"Solver" : "DG",
	"SolutionOrder" : 1,
	"SolutionBasis" : "LagrangeQuad",
	"DiffFluxSwitch" : True,
}

nelem = 4 
num_elem_x = nelem
num_elem_y = nelem

#Mesh = {
#	"File" : "box_4.msh",
#}
Mesh = {
	"ElementShape" : "Quadrilateral",
	"NumElemsX" : num_elem_x,
	"NumElemsY" : num_elem_y,
	"xmin" : 0.,
	"xmax" : 10.,
	"ymin" : 0.,
	"ymax" : 10.,
	"PeriodicBoundariesX" : ["x2", "x1"],
	"PeriodicBoundariesY" : ["y1", "y2"],
}

Physics = {
	"Type" : "NavierStokes",
	"ConvFluxNumerical" : "Roe",
	"DiffFluxNumerical" : "SIP",
	"GasConstant" : 0.4,
}

InitialCondition = {
	"Function" : "ManufacturedSolution",
}

ExactSolution = InitialCondition.copy()

#BoundaryConditions = {
#        "y1" : {
#                "BCType" : "SlipWall",
#        },
#        "y2" : {
#                "BCType" : "SlipWall",
#        },
#        "x2" : {
#                "BCType" : "SlipWall",
#        },
#        "x1" : {
#                "BCType" : "SlipWall",
#        }
#}

SourceTerms = {
	"Source1" : { # Name of source term ("Source1") doesn't matter
		"Function" : "ManufacturedSource",
	},
}

Output = {
	"Prefix" : "Data",
	"WriteInterval" : 100,
	"WriteInitialSolution" : True,
	"WriteFinalSolution" : True,
	"AutoPostProcess" : True,
}
