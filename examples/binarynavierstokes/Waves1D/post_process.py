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
### Temperature
plot.prepare_plot()
# DG solution
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, 
		plot_exact=False, plot_IC=False, create_new_figure=True, 
		fmt='bo', legend_label="DG")
# Initial condition
plot.plot_solution(mesh, physics, solver, "Pressure", plot_IC=True, 
		plot_numerical=False, create_new_figure=False, fmt='k--')

plot.show_plot()

### Temperature
plot.prepare_plot()
# DG solution
plot.plot_solution(mesh, physics, solver, "Temperature", plot_numerical=True, 
		plot_exact=False, plot_IC=False, create_new_figure=True, 
		fmt='bo', legend_label="DG")
# Initial condition
plot.plot_solution(mesh, physics, solver, "Temperature", plot_IC=True, 
		plot_numerical=False, create_new_figure=False, fmt='k--')

plot.show_plot()

### Temperature
plot.prepare_plot()
# DG solution
plot.plot_solution(mesh, physics, solver, "MassFraction", plot_numerical=True, 
		plot_exact=False, plot_IC=False, create_new_figure=True, 
		fmt='bo', legend_label="DG")
# Initial condition
plot.plot_solution(mesh, physics, solver, "MassFraction", plot_IC=True, 
		plot_numerical=False, create_new_figure=False, fmt='k--')

plot.show_plot()

### Temperature
plot.prepare_plot()
# DG solution
plot.plot_solution(mesh, physics, solver, "Velocity", plot_numerical=True, 
		plot_exact=False, plot_IC=False, create_new_figure=True, 
		fmt='bo', legend_label="DG")
# Initial condition
plot.plot_solution(mesh, physics, solver, "Velocity", plot_IC=True, 
		plot_numerical=False, create_new_figure=False, fmt='k--')

plot.show_plot()

