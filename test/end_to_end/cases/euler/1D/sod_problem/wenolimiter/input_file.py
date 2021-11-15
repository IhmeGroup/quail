import numpy as np
import copy

FinalTime = 0.25
NumTimeSteps = 1500

TimeStepping = {
    "InitialTime" : 0.,
    "FinalTime" : FinalTime,
    "CFL" : 0.2,
    "TimeStepper" : "SSPRK3",
}


Numerics = {
    "SolutionOrder" : 1,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ApplyLimiters" : "WENO",
    "ShockIndicator" : "MinMod",
    "TVBParameter" : 0.1,
    "NodeType" : "Equidistant",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
}

Output = {
    "AutoPostProcess" : True,
    "Prefix" : "Data",
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 100,
    "xmin" : -5.,
    "xmax" : 5.,
}


Physics = {
    "Type" : "Euler",
    "ConvFluxNumerical" : "Roe",
    "GasConstant" : 1.,
    "SpecificHeatRatio" : 1.4,
}


state = { 
    "Function" : "RiemannProblem",
    "rhoL" : 1., 
    "uL" : 0.,
    "pL" : 1.,
    "rhoR" : 0.125, 
    "uR" : 0.,
    "pR" : 0.1,
    "xd" : 0.0,
}

InitialCondition = state
ExactSolution = state

BoundaryConditions = {
    "x1" : {
        "BCType" : "SlipWall"
        },
    "x2" : { 
        "BCType" : "SlipWall"
        }
}

Output = {
	"AutoPostProcess" : False,
	"ProgressBar" : True,
}

