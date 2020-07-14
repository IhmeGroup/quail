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

TotErr, _ = post.L2_error(mesh, physics, solver, "Pressure")
# plot
axis = None
# axis = [-5., 5., -5., 5.]
plot.PreparePlot(axis=axis, linewidth=0.5)
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='bo', legend_label="DG", equidistant_pts=True, 
			include_mesh=True, regular_2D=True, equal_AR=False, show_elem_IDs=True)
# # plot.PlotSolution(mesh, physics, solver, "Density", Equidistant=True, PlotExact=False, include_mesh=True, 
# # 	Regular2D=True, show_triangulation=False, show_elem_IDs=True)
plot.SaveFigure(FileName='Pressure', FileType='pdf', CropLevel=2)
#plot.PreparePlot(close_all=False, linewidth=1.5)
# plot.plot_line_probe(mesh, physics, solver, "Density", xy1=[-5.,1.], xy2=[5.,1.], nPoint=101, PlotExact=True, PlotIC=True)
# plot.plot_line_probe(mesh, physics, solver, "Density", xy1=[-5.,1.], xy2=[5.,1.], nPoint=101, plot_numerical=False, plot_exact=False,
# 		plot_IC=True, create_new_figure=True, ylabel=None, vs_x=True, fmt="k-.", legend_label=None)
# plot.plot_line_probe(mesh, physics, solver, "Density", xy1=[-5.,1.], xy2=[5.,1.], nPoint=101, plot_numerical=False, plot_exact=True,
# 		plot_IC=False, create_new_figure=False, fmt="k-", legend_label=None)
# plot.plot_line_probe(mesh, physics, solver, "Density", xy1=[-5.,1.], xy2=[5.,1.], nPoint=101, plot_numerical=True, plot_exact=False,
# 		plot_IC=False, create_new_figure=False, fmt="bo", legend_label=None)
# plot.SaveFigure(FileName='line', FileType='pdf', CropLevel=2)
plot.ShowPlot()
