import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

# Read data file
fname = "Data_pp.pkl"
solver = readwritedatafiles.read_data_file(fname)

# Unpack
mesh = solver.mesh
physics = solver.physics

''' Plot '''
### Density
plot.prepare_plot()
# Exact solution
plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=False,
		plot_exact=True, plot_IC=False, create_new_figure=True,
		fmt='k-.', legend_label="Exact")
# DG solution
plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=True,
		plot_exact=False, plot_IC=False, plot_average=False,
		create_new_figure=False, fmt='r.', legend_label="Positivity Preserving")



# Read data file
fname = "Data_av.pkl"
solver = readwritedatafiles.read_data_file(fname)

# Unpack
mesh = solver.mesh
physics = solver.physics

# DG solution
plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=True,
		plot_exact=False, plot_IC=False, plot_average=False,
		create_new_figure=False, fmt='b.', legend_label="Artificial Viscosity")


plot.save_figure(file_name='Density', file_type='svg')


plot.show_plot()
