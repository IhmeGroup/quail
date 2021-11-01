import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

# Read data file
fname = "Data_final.pkl"
solver = readwritedatafiles.read_data_file(fname)

# Unpack
mesh = solver.mesh
physics = solver.physics

# Compute L2 error
#post.get_error(mesh, physics, solver, "Density")

''' Plot '''
# Density contour
plot.prepare_plot(linewidth=0.5)
plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=True,
		plot_exact=False, plot_IC=False, create_new_figure=True, fmt='bo',
		legend_label="DG", include_mesh=True, regular_2D=False,
		show_elem_IDs=False, ignore_colorbar=True, equal_AR=True)
plot.save_figure(file_name='flow', file_type='svg', crop_level=2)

### Line probe (y = 1) ###
plot.prepare_plot(close_all=False, linewidth=1.5)
# Parameters
xy1 = [-5.,1.]; xy2 = [5.,1.]
# Initial condition
plot.plot_line_probe(mesh, physics, solver, "Density", xy1=xy1, xy2=xy2,
		plot_numerical=False, plot_exact=False, plot_IC=True,
		create_new_figure=True, ylabel=None, vs_x=True, fmt="k-.",
		legend_label=None)
# Exact solution
#plot.plot_line_probe(mesh, physics, solver, "Density", xy1=xy1, xy2=xy2,
#		plot_numerical=False, plot_exact=True, plot_IC=False,
#		create_new_figure=False, fmt="k-", legend_label=None)
# DG solution
plot.plot_line_probe(mesh, physics, solver, "Density", xy1=xy1, xy2=xy2,
		plot_numerical=True, plot_exact=False, plot_IC=False,
		create_new_figure=False, fmt="bo", legend_label=None)
# Save figure
plot.save_figure(file_name='line', file_type='pdf', crop_level=2)

plot.show_plot()
