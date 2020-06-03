import sys; sys.path.append('../../../../src')
import code
import numpy as np
import os 

import processing.post as Post
import processing.plot as Plot

import driver
import general

CurrentDir = os.path.dirname(os.path.abspath(__file__)) + "/"


solver, EqnSet, mesh = driver.driver()

### Postprocess
# Error
# TotErr,_ = Post.L2_error(mesh, EqnSet, solver, "Scalar")
# Plot
Plot.PreparePlot()
Plot.PlotSolution(mesh, EqnSet, solver, "Scalar", PlotExact=False, PlotIC=True, Label="u")
Plot.ShowPlot()