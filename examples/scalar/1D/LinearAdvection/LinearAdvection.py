import sys; sys.path.append('../../../../src')
import code
import numpy as np
import os 

import processing.post as Post
import processing.plot as Plot

import driver
import general

CurrentDir = os.path.dirname(os.path.abspath(__file__)) + "/"


TimeStepping = {
    "StartTime" : 0.,
    "EndTime" : 0.5,
    "nTimeStep" : 40,
    "TimeScheme" : "RK4",
}

Numerics = {
    "InterpOrder" : 2,
    "InterpBasis" : "LagrangeEqSeg",
    "Solver" : "DG",
}

Output = {}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "nElem_x" : 16,
    "xmin" : -1.,
    "xmax" : 1.,
}

Physics = {
    "Type" : "ConstAdvScalar",
    "ConvFlux" : "LaxFriedrichs",
    "ConstVelocity" : 1.,
}

InitialCondition = {
    "Function" : "Sine",
    "omega" : 2*np.pi,
    "SetAsExact" : True,
}

BoundaryConditions = {
    "Left" : {
	    "Function" : "Sine",
	    "omega" : 2*np.pi,
    	"BCType" : "FullState",
    },
    "Right" : {
    	"Function" : None,
    	"BCType" : "Extrapolation",
    },
}


solver, EqnSet, mesh = driver.driver(TimeStepping, Numerics, Output, Mesh,
		Physics, InitialCondition, BoundaryConditions)

### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, EqnSet, solver, "Scalar")
# Plot
Plot.PreparePlot()
Plot.PlotSolution(mesh, EqnSet, solver, "Scalar", PlotExact=True, PlotIC=True, Label="u")
Plot.ShowPlot()