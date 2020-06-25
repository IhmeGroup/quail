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

TotErr, _ = post.L2_error(mesh, physics, solver, "Density")
# plot
axis = None
# axis = [-5., 5., -5., 5.]
plot.PreparePlot(axis=axis, linewidth=0.5)
plot.PlotSolution(mesh, physics, solver, "Density", Equidistant=True, PlotExact=False, include_mesh=True, 
	Regular2D=True, ShowTriangulation=False, show_elem_IDs=True)
plot.SaveFigure(FileName='vortex', FileType='pdf', CropLevel=2)
plot.PreparePlot(close_all=False, linewidth=1.5)
plot.plot_line_probe(mesh, physics, solver, "Density", xy1=[-5.,1.], xy2=[5.,1.], nPoint=101, PlotExact=True, PlotIC=True)
plot.SaveFigure(FileName='line', FileType='pdf', CropLevel=2)
plot.ShowPlot()
