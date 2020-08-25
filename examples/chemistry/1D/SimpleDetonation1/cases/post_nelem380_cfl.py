import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

### Postprocess
# Note: This case took ~10 minutes to run
fname = "Data_cflp025_nelem380.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)

mesh = solver.mesh
physics = solver.physics


## Note: This case took ~40 minutes to run.
fname = "Data_cflp005_nelem360.pkl"
solver4 = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)

# Unpack
mesh4= solver4.mesh
physics4 = solver4.physics
# TotErr,_ = post.get_error(mesh, physics, solver, "Density")
# Plot
plot.prepare_plot()


plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='b', legend_label="cfl=0.025", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False)

plot.plot_solution(mesh4, physics4, solver4, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='m', legend_label="nelem=0.005", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False)
plot.plot_solution(mesh, physics, solver, "Density", plot_exact=True, plot_numerical=False, create_new_figure=False, fmt='k-')
plot.plot_solution(mesh, physics, solver, "Density", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')

plot.show_plot()
