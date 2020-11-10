import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

### Postprocess
fname = "Data_final.pkl"
solver = readwritedatafiles.read_data_file(fname)

# Unpack
mesh = solver.mesh
physics = solver.physics

### Postprocess
plot.prepare_plot()

# Density
plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=False, 
			plot_exact=True, plot_IC=False, create_new_figure=True, 
			fmt='k-.', legend_label="Exact")

plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=True, 
			plot_exact=False, plot_IC=False, create_new_figure=False,
            fmt='bo', legend_label="Numerical")

# Pressure
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=False, 
			plot_exact=True, plot_IC=False, create_new_figure=True, 
			fmt='k-.', legend_label="Exact")

plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, 
			plot_exact=False, plot_IC=False, create_new_figure=False,
            fmt='bo', legend_label="Numerical")

plot.show_plot()