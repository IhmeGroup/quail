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
### Total energy
plot.prepare_plot()
# DG solution
plot.plot_solution(mesh, physics, solver, "Energy", plot_numerical=True, 
		plot_exact=False, plot_IC=False, create_new_figure=True, 
		ylabel=None, fmt='bo', legend_label="DG")
# Initial condition
plot.plot_solution(mesh, physics, solver, "Energy", plot_numerical=False, 
		plot_exact=False, plot_IC=True, create_new_figure=False, 
		ylabel=None, fmt='k--', legend_label="DG")

plot.show_plot()
