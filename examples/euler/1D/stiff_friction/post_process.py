import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

### Postprocess
fname = "Data_final.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)

# Unpack
mesh = solver.mesh
physics = solver.physics

# Plot
plot.prepare_plot()

plot.plot_solution(mesh, physics, solver, "Energy", plot_numerical=True, 
			plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='bo', legend_label="DG")
plot.plot_solution(mesh, physics, solver, "Energy", plot_numerical=False, 
			plot_exact=False, plot_IC=True, create_new_figure=False, 
			ylabel=None, fmt='k--', legend_label="DG")

plot.show_plot()
