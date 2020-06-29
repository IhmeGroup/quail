import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

### Postprocess
fname = "Data_final.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.Time)

# Unpack
mesh = solver.mesh
physics = solver.EqnSet

# Error
# TotErr,_ = post.L2_error(mesh, physics, solver, "Density")
# Plot
plot.PreparePlot()
plot.PlotSolution(mesh, physics, solver, "Energy", PlotExact=False, PlotIC = True)
plot.PlotSolution(mesh, physics, solver, "XMomentum", PlotExact=False, PlotIC = True)
plot.PlotSolution(mesh, physics, solver, "Density", PlotExact=False, PlotIC = True)


plot.SaveFigure(FileName='StiffFriction', FileType='pdf', CropLevel=2)

plot.ShowPlot()
