n_elem = 1

CFL = 0.0025
dx = 1./n_elem
u = 1.
dt = CFL*dx/u
tfinal = 0.4
nt = int(tfinal/dt + 1.e-12)

TimeStepping = {
    "EndTime" : tfinal,
    "num_time_steps" : nt,
    "TimeScheme" : "FE",
}

Numerics = {
    "InterpOrder" : 8,
    # "InterpBasis" : "LagrangeEqQuad",
    "InterpBasis" : "LegendreQuad",
    # "InterpBasis" : "LagrangeEqTri",
    # "ElementQuadrature" : "Dunavant",
    # "FaceQuadrature" : "GaussLegendre",
}

Mesh = {
    "ElementShape" : "Quadrilateral",
     # "ElementShape" : "Triangle",
    "NumElemsX" : n_elem,
    "NumElemsY" : n_elem,
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
    # "GasConstant" : 1.,
}

InitialCondition = {
    "Function" : "TaylorGreenVortex",
    # "State" : [1.0],
    # "SetAsExact" : False,
}

ExactSolution = InitialCondition.copy()

SourceTerms = {
    "Source1" : {
        "Function" : "TaylorGreenSource",
    },
}

Output = {
    "AutoPostProcess" : True,
}
