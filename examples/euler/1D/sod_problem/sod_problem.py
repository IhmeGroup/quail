import numpy as np
import copy

FinalTime = 0.25
NumTimeSteps = 1500

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "CFL" : 0.1,
    "TimeStepper" : "SSPRK3",
}


Numerics = {
    "SolutionOrder" : 2,
    "SolutionBasis" : "LagrangeQuad",
    "Solver" : "DG",
    "ApplyLimiters" : "WENOquad",
    #"ApplyLimiters" : "PositivityPreserving",
    "ShockIndicator" : "MinModQuad",
    "TVBParameter" : 120,
    "NodeType" : "Equidistant",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
}

Output = {
    "AutoPostProcess" : False,
    "Prefix" : "Data",
}

Mesh = {
    "File" : None,
    "ElementShape" : "Quadrilateral",
    "NumElemsX" : 60,
    "NumElemsY" : 60,
    "xmin" : -0.5,
    "xmax" : 0.5,
    "ymin" : -0.5,
    "ymax" : 0.5,
}


Physics = {
    "Type" : "Euler",
    "ConvFluxNumerical" : "Roe",
    "GasConstant" : 1.,
    "SpecificHeatRatio" : 1.4,
}


state = {
    "Function" : "Riemann_2D",
}

InitialCondition = state
#ExactSolution = state

BoundaryConditions = {
    "x1" : {
        "BCType" : "SlipWall"
        },
    "x2" : {
        "BCType" : "SlipWall"
        },
    "y1" : {
        "BCType" : "SlipWall"
        },
    "y2" : {
        "BCType" : "SlipWall"
        }
}
