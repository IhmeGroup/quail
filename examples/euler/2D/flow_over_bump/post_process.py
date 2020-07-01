import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

### Postprocess
fname = "p2_final.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.Time)

# Unpack
mesh = solver.mesh
physics = solver.EqnSet

TotErr, _ = post.L2_error(mesh, physics, solver, "Entropy", NormalizeByVolume=False)
# Plot
axis = None
EqualAR = False
# axis = [-5., 5., -5., 5.]
plot.PreparePlot(axis=axis, linewidth=0.5)
# plot.PlotSolution(mesh, physics, solver, "Pressure", Equidistant=True, PlotExact=False, include_mesh=True, 
# 	show_triangulation=False, EqualAR=EqualAR, show_elem_IDs=True)
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, create_new_figure=True, 
			include_mesh=True, regular_2D=False, equal_AR=False, show_elem_IDs=True)
plot.SaveFigure(FileName='Pressure', FileType='pdf', CropLevel=2)
# plot.PlotSolution(mesh, physics, solver, "Entropy", Equidistant=True, PlotExact=False, include_mesh=True, 
# 	show_triangulation=False, EqualAR=EqualAR)
plot.plot_solution(mesh, physics, solver, "Entropy", plot_numerical=True, create_new_figure=True, 
			include_mesh=True, regular_2D=False)
plot.SaveFigure(FileName='Entropy', FileType='pdf', CropLevel=2)

post.get_boundary_info(mesh, physics, solver, "bottom", "Pressure", integrate=True, 
		vec=[1.,0.], dot_normal_with_vec=True, plot_vs_x=True, plot_vs_y=False, fmt="bo", ylabel="$F_x$")
plot.ShowPlot()
