import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

# Read data file
fname = "p2_final.pkl"
solver = readwritedatafiles.read_data_file(fname)

# Unpack
mesh = solver.mesh
physics = solver.physics

# Compute L2 error
post.get_error(mesh, physics, solver, "Entropy", 
		normalize_by_volume=False)

''' Plot '''
### Pressure contour ###
plot.prepare_plot(linewidth=0.5)
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, 
		create_new_figure=True, include_mesh=True, regular_2D=False, 
		show_elem_IDs=True)
# Save figure
plot.save_figure(file_name='Pressure', file_type='pdf', crop_level=2)

### Entropy contour ###
plot.plot_solution(mesh, physics, solver, "Entropy", plot_numerical=True, 
		create_new_figure=True, include_mesh=True, regular_2D=False)
# Save figure
plot.save_figure(file_name='Entropy', file_type='pdf', crop_level=2)

### Boundary info ###
# Plot pressure in x-direction along wall
# Boundary integral gives drag force in x-direction
post.get_boundary_info(solver, mesh, physics, "bottom", "Pressure", 
		dot_normal_with_vec=True, vec=[1.,0.], integrate=True, 
		plot_vs_x=True, plot_vs_y=False, fmt="bo", ylabel="$F_x$")

plot.show_plot()
