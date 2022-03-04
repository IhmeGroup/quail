TimeStepping = {
	"FinalTime" : 10,
	"TimeStepSize" : 0.05,
	"TimeStepper" : "LSRK4",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeQuad",
	"ConvFluxSwitch" : False,
}

nelem = 2; 

Mesh = {
	"ElementShape" : "Quadrilateral",
	"NumElemsX" : nelem,
	"NumElemsY" : nelem,
	"xmin" : -1.,
	"xmax" : 1.,
	"ymin" : -1.,
	"ymax" : 1.,
	"PeriodicBoundariesX" : ["x2", "x1"],
	"PeriodicBoundariesY" : ["y2", "y1"], 
}

Physics = {
	"Type" : "Wildfire",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"Transport" : "Constant",
	# "Viscosity" : 1e-5,
	# "PrandtlNumber" :  0.71, 
	# "SpecificHeatRatio" : 1.4,
	# "GasConstant" : 287, 
}

InitialCondition = {
	"Function" : "WildfireBurn",
}

# ExactSolution = InitialCondition.copy()

# d = {
# 		"BCType" : "StateAll",
# 		"Function" : "IsentropicVortex",
# }

# BoundaryConditions = {
# 	"y1" : d,
# 	"y2" : d,
# }

SourceTerms = {
	"Wildfire Source Terms" : {
		"Function" : "WildfireSource",  
	},

}

Output = {
	"AutoPostProcess" : False,
	"Verbose" : True,
}
