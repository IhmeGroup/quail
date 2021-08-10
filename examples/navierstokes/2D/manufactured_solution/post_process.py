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
post.get_error(mesh, physics, solver, "Pressure")

''' Plot '''
# Pressure contour
plot.prepare_plot(linewidth=0.5)
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, 
		plot_exact=False, plot_IC=False, create_new_figure=True, 
		fmt='bo', legend_label="DG", include_mesh=True, regular_2D=True, 
		equal_AR=False)
# Save figure
plot.save_figure(file_name='Pressure', file_type='pdf', crop_level=2)

plot.show_plot()
