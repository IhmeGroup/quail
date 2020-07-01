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

solver.Time = 0.
TotErr,_ = post.L2_error(mesh, physics, solver, "Scalar")
# Plot
axis = None
# axis = [-5., 5., -5., 5.]
plot.PreparePlot(axis=axis, linewidth=0.5)
# plot.PlotSolution(mesh, physics, solver, "Scalar", Equidistant=True, PlotExact=False, include_mesh=True, 
# 	Regular2D=True, show_triangulation=False)
plot.plot_solution(mesh, physics, solver, "Scalar", plot_numerical=True, create_new_figure=True, 
			include_mesh=True, regular_2D=True, equal_AR=False, show_elem_IDs=True)
plot.SaveFigure(FileName='Gaussian', FileType='pdf', CropLevel=2)

# plot.plot_line_probe(mesh, physics, solver, "Scalar", xy1=[-5.,-5.], xy2=[5.,5.], nPoint=101, PlotExact=True, PlotIC=True)
plot.plot_line_probe(mesh, physics, solver, "Scalar", xy1=[-5.,-5.], xy2=[5.,5.], nPoint=101, plot_numerical=False, plot_exact=False,
		plot_IC=True, create_new_figure=True, ylabel=None, vs_x=True, fmt="k-.", legend_label=None)
plot.plot_line_probe(mesh, physics, solver, "Scalar", xy1=[-5.,-5.], xy2=[5.,5.], nPoint=101, plot_numerical=False, plot_exact=True,
		plot_IC=False, create_new_figure=False, fmt="k-", legend_label=None)
plot.plot_line_probe(mesh, physics, solver, "Scalar", xy1=[-5.,-5.], xy2=[5.,5.], nPoint=101, plot_numerical=True, plot_exact=False,
		plot_IC=False, create_new_figure=False, fmt="bo", legend_label=None)
# Post.get_boundary_info(mesh, EqnSet, solver, "y1", "Scalar", integrate=True, 
# 		vec=[0.,1.], dot_normal_with_vec=True, plot_vs_x=True, plot_vs_y=False)
plot.ShowPlot()
