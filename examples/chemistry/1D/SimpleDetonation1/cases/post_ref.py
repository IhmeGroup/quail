import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

### Postprocess
fname = "Data_cflp005_nelem45.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)

# Unpack
mesh = solver.mesh
physics = solver.physics

fname = "Data_cflp005_nelem90.pkl"
solver2 = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)

# Unpack
mesh2 = solver2.mesh
physics2 = solver2.physics


# Note: This case took ~10 minutes to run.
fname = "Data_cflp005_nelem180.pkl"
solver3 = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)

# Unpack
mesh3 = solver3.mesh
physics3 = solver3.physics

## Note: This case took ~40 minutes to run.
fname = "Data_cflp005_nelem360.pkl"
solver4 = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)

# Unpack
mesh4= solver4.mesh
physics4 = solver4.physics
# TotErr,_ = post.L2_error(mesh, physics, solver, "Density")
# Plot
plot.PreparePlot()


plot.plot_solution(mesh, physics, solver, "MassFraction", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='b', legend_label="nelem=45", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False)


plot.plot_solution(mesh2, physics2, solver2, "MassFraction", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='r', legend_label="nelem=90", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False)
plot.plot_solution(mesh3, physics3, solver3, "MassFraction", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='g', legend_label="nelem=180", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False)

plot.plot_solution(mesh4, physics4, solver4, "MassFraction", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='m', legend_label="nelem=360", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False)
plot.plot_solution(mesh, physics, solver, "MassFraction", plot_exact=True, plot_numerical=False, create_new_figure=False, fmt='k-')
plot.plot_solution(mesh, physics, solver, "MassFraction", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')

plot.ShowPlot()
