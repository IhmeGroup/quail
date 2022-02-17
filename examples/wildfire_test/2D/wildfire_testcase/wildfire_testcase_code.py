TimeStepping = {
	"FinalTime" : 1.0,
	"TimeStepSize" : 0.05,
	"TimeStepper" : "LSRK4",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeTri",
	"ElementQuadrature" : "Dunavant",
	"FaceQuadrature" : "GaussLegendre",
}

Mesh = {
	"ElementShape" : "Triangle",
	"NumElemsX" : 5,
	"NumElemsY" : 5,
	"xmin" : -5.,
	"xmax" : 5.,
	"ymin" : -5.,
	"ymax" : 5.,
	# "PeriodicBoundariesX" : ["x2", "x1"],
}

Physics = {
	"Type" : "Wildfire",
	"ConvFluxNumerical" : "LaxFriedrichs",
	# "GasConstant" : 1.,
}

InitialCondition = {
	"Function" : "WildfireBurn",
}

# ExactSolution = InitialCondition.copy()

# d = {
# 		"BCType" : "StateAll",
# 		"Function" : "IsentropicVortex",
# }

BoundaryConditions = {
	"y1" : d,
	"y2" : d,
}

SourceTerms = {
	"Wood Density Source" : {
		"Function" : "WildfireSource", 
		"Nwood" : 1., 
		"Fwood" : 1.,  
	},

	"Water Density Source" : {
		"Function" : "WildfireSource", 
		"Fwater" : 1., 
		
	},

	"Temperature Source" : {
		"Function" : "WildfireSource",
		"Z" : 1., 
		
	},
}

Output = {
	"AutoPostProcess" : True,
	"Verbose" : True,
}
