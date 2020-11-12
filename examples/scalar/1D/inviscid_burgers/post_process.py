import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

### Postprocess
plot.prepare_plot()


# fname = "Ref_final.pkl"
# solver = readwritedatafiles.read_data_file(fname)
# print('Solution Final Time:', solver.time)

# # Unpack
# mesh = solver.mesh
# physics = solver.physics
# plot.plot_solution(mesh, physics, solver, "Scalar", plot_numerical=True,
#			plot_exact=False, plot_IC=False, create_new_figure=True,
# 			ylabel=None, fmt='k-', legend_label="Ref", equidistant_pts=True,
# 			include_mesh=False, regular_2D=False, equal_AR=False)
#
# plot.plot_solution(mesh, physics, solver, "Scalar", plot_IC=True,
#			plot_numerical=False, create_new_figure=False, fmt='k--')

fname = "Data_final.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)

# Unpack
mesh = solver.mesh
physics = solver.physics
plot.plot_solution(mesh, physics, solver, "Scalar", plot_numerical=True,
			plot_exact=False, plot_IC=False, create_new_figure=True,
			ylabel=None, fmt='go', legend_label="No Limiter",
			equidistant_pts=True, include_mesh=False, regular_2D=False,
			equal_AR=False)

# Compute L2 error
TotErr, _ = post.get_error(mesh, physics, solver, "Scalar")

plot.plot_solution(mesh, physics, solver, "Scalar", plot_exact=True,
		plot_numerical=False, create_new_figure=False, fmt='k-')

fname = "Data_final.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)

# Unpack
# mesh = solver.mesh
# physics = solver.physics

# plot.plot_solution(mesh, physics, solver, "Scalar", plot_numerical=True,
# 			plot_exact=False, plot_IC=False, create_new_figure=False,
# 			ylabel=None, fmt='bo', legend_label="WENO Limiter", equidistant_pts=True,
# 			include_mesh=False, regular_2D=False, equal_AR=False)

# plot.save_figure(file_name='inv_burgers_weno_testing', file_type='pdf', crop_level=2)

plot.show_plot()
