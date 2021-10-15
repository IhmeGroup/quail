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
post.get_error( mesh, physics, solver, "Scalar")

import code; code.interact(local=locals())
''' Plot '''
# Scalar contour
plot.prepare_plot(linewidth=0.5)
plot.plot_solution(mesh, physics, solver, "Scalar", plot_numerical=True, 
		create_new_figure=True, include_mesh=True, regular_2D=True, 
		equal_AR=False)
plot.plot_solution(mesh, physics, solver, "Scalar", plot_numerical=False,
		plot_exact=True, create_new_figure=True, include_mesh=True,
		regular_2D=True, equal_AR=False)
# Save figure
#plot.save_figure(file_name='gaussian', file_type='pdf', crop_level=2)

plot.show_plot()
