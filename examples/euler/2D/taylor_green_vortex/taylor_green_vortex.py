TimeStepping = {
    "FinalTime" : 0.4,
    "NumTimeSteps" : 160,
    "TimeStepper" : "FE",
}

Numerics = {
    "SolutionOrder" : 8,
    "SolutionBasis" : "LegendreQuad",
}

num_elem_x = 1
Mesh = {
    "ElementShape" : "Quadrilateral",
    "NumElemsX" : num_elem_x,
    "NumElemsY" : num_elem_x,
    "xmin" : 0.,
    "xmax" : 1.,
    "ymin" : 0.,
    "ymax" : 1.,
    "PeriodicBoundariesX" : ["x2", "x1"],
    "PeriodicBoundariesY" : ["y1", "y2"],
}

Physics = {
    "Type" : "Euler",
    "ConvFluxNumerical" : "Roe",
}

InitialCondition = {
    "Function" : "TaylorGreenVortex",
}

ExactSolution = InitialCondition.copy()

SourceTerms = {
    "Source1" : { # Name of source term ("Source1") doesn't matter
        "Function" : "TaylorGreenSource",
    },
}

Output = {
    "AutoPostProcess" : True,
}
