import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

# Read data file
fname = "Data_final.pkl"
solver = readwritedatafiles.read_data_file(fname)

# Unpack
mesh = solver.mesh
physics = solver.physics

''' Plot '''
### Line probe ###
plot.prepare_plot(linewidth=1.5)
# Parameters
xy1 = [0., 0.5]; xy2 = [1., 0.5]

# DG solution
plot.plot_line_probe(mesh, physics, solver, "Pressure", xy1=xy1, xy2=xy2, 
		plot_numerical=True, plot_exact=False, plot_IC=False, 
		create_new_figure=True, fmt="bo", legend_label="DG")

plot.plot_line_probe(mesh, physics, solver, "Pressure", xy1=xy1, xy2=xy2, 
		plot_numerical=False, plot_exact=False, plot_IC=True, 
		create_new_figure=False, fmt="k--")

plot.show_plot()

plot.plot_line_probe(mesh, physics, solver, "Temperature", xy1=xy1, xy2=xy2, 
		plot_numerical=True, plot_exact=False, plot_IC=False, 
		create_new_figure=True, fmt="bo", legend_label="DG")

plot.plot_line_probe(mesh, physics, solver, "Temperature", xy1=xy1, xy2=xy2, 
		plot_numerical=False, plot_exact=False, plot_IC=True, 
		create_new_figure=False, fmt="k--")

plot.show_plot()

plot.plot_line_probe(mesh, physics, solver, "MassFraction", xy1=xy1, xy2=xy2, 
		plot_numerical=True, plot_exact=False, plot_IC=False, 
		create_new_figure=True, fmt="bo", legend_label="DG")

plot.plot_line_probe(mesh, physics, solver, "MassFraction", xy1=xy1, xy2=xy2, 
		plot_numerical=False, plot_exact=False, plot_IC=True, 
		create_new_figure=False, fmt="k--")

plot.show_plot()

plot.plot_line_probe(mesh, physics, solver, "XVelocity", xy1=xy1, xy2=xy2, 
		plot_numerical=True, plot_exact=False, plot_IC=False, 
		create_new_figure=True, fmt="bo", legend_label="DG")

plot.plot_line_probe(mesh, physics, solver, "XVelocity", xy1=xy1, xy2=xy2, 
		plot_numerical=False, plot_exact=False, plot_IC=True, 
		create_new_figure=False, fmt="k--")

plot.show_plot()

plot.prepare_plot(linewidth=0.5)
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=False, 
		plot_exact=False, plot_IC=True, create_new_figure=True, 
		fmt='bo', legend_label="DG", include_mesh=True, regular_2D=True, 
		equal_AR=False, show_elem_IDs=False)

plot.show_plot()

plot.prepare_plot(linewidth=0.5)
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, 
		plot_exact=False, plot_IC=False, create_new_figure=True, 
		fmt='bo', legend_label="DG", include_mesh=True, regular_2D=True, 
		equal_AR=False, show_elem_IDs=False)

plot.show_plot()

