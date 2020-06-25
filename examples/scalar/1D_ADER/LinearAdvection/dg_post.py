import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

### Postprocess
fname = "dg_final.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.Time)

# Unpack
mesh = solver.mesh
physics = solver.EqnSet

# Error
TotErr, _ = post.L2_error(mesh, physics, solver, "Scalar")
# Plot
plot.PreparePlot()
plot.PlotSolution(mesh, physics, solver, "Scalar", PlotExact=True, PlotIC=True, Label="u")

plot.SaveFigure(FileName='LinearAdvection', FileType='pdf', CropLevel=2)

plot.ShowPlot()
