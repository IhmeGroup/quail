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
### Density
plot.prepare_plot()
# Exact solution
plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=False, 
		plot_exact=True, plot_IC=False, create_new_figure=True, 
		fmt='k-.', legend_label="Exact")
# DG solution
plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=True, 
		plot_exact=False, plot_IC=False, create_new_figure=False,
        fmt='bo', legend_label="Numerical")

### Pressure
# Exact solution
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=False, 
		plot_exact=True, plot_IC=False, create_new_figure=True, 
		fmt='k-.', legend_label="Exact")
# DG solution
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, 
		plot_exact=False, plot_IC=False, create_new_figure=False,
        fmt='bo', legend_label="Numerical")

plot.show_plot()