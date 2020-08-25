import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

### Postprocess
fname = "p2_final.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)

# Unpack
mesh = solver.mesh
physics = solver.physics

TotErr, _ = post.get_error(mesh, physics, solver, "Entropy", normalize_by_volume=False)
# Plot
axis = None
equal_AR = False
# axis = [-5., 5., -5., 5.]
plot.prepare_plot(axis=axis, linewidth=0.5)
# plot.PlotSolution(mesh, physics, solver, "Pressure", Equidistant=True, PlotExact=False, include_mesh=True, 
# 	show_triangulation=False, equal_AR=equal_AR, show_elem_IDs=True)
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, create_new_figure=True, 
			include_mesh=True, regular_2D=False, equal_AR=False, show_elem_IDs=True)
plot.save_figure(file_name='Pressure', FileType='pdf', CropLevel=2)
# plot.PlotSolution(mesh, physics, solver, "Entropy", Equidistant=True, PlotExact=False, include_mesh=True, 
# 	show_triangulation=False, equal_AR=equal_AR)
plot.plot_solution(mesh, physics, solver, "Entropy", plot_numerical=True, create_new_figure=True, 
			include_mesh=True, regular_2D=False)
plot.save_figure(file_name='Entropy', FileType='pdf', CropLevel=2)

post.get_boundary_info(solver, mesh, physics, "bottom", "Pressure", dot_normal_with_vec=True, vec=[1.,0.], 
		integrate=True, plot_vs_x=True, plot_vs_y=False, fmt="bo", ylabel="$F_x$")
plot.show_plot()
