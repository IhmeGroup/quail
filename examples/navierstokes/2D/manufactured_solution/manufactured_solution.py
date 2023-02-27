TimeStepping = {
	"FinalTime" : 1e-2,
	"TimeStepSize" : 0.0001,
	"TimeStepper" : "SSPRK3",
}

Numerics = {
	"Solver" : "DG",
	"SolutionOrder" : 3,
	"SolutionBasis" : "LagrangeQuad",
}

nelem = 4 
num_elem_x = nelem
num_elem_y = nelem

Mesh = {
	"ElementShape" : "Quadrilateral",
	"NumElemsX" : num_elem_x,
	"NumElemsY" : num_elem_y,
	"xmin" : 0.,
	"xmax" : 1.,
	"ymin" : 0.,
	"ymax" : 1.,
}

Physics = {
	"Type" : "NavierStokes",
	"ConvFluxNumerical" : "Roe",
	"DiffFluxNumerical" : "SIP",
	"GasConstant" : 1.0,
	"Transport" : "Constant",
	"Viscosity" : 1e-1,
	"PrandtlNumber" : 0.71,
}

InitialCondition = {
	"Function" : "ManufacturedSolution",
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
       "y1" : {
               "BCType" : "StateAll",
               "Function" : "ManufacturedSolution",
       },
       "y2" : {
               "BCType" : "StateAll",
               "Function" : "ManufacturedSolution",
       },
       "x2" : {
               "BCType" : "StateAll",
               "Function" : "ManufacturedSolution",
       },
       "x1" : {
               "BCType" : "StateAll",
               "Function" : "ManufacturedSolution",
       }
}

SourceTerms = {
	"Source1" : { # Name of source term ("Source1") doesn't matter
		"Function" : "ManufacturedSource",
	},
}

Output = {
	"Prefix" : "Data",
	#"WriteInterval" : 100,
	#"WriteInitialSolution" : True,
	"WriteFinalSolution" : True,
	"AutoPostProcess" : True,
}
