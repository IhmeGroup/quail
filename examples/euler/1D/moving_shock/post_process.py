import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

### Postprocess
fname = "Data_final.pkl"
solver = readwritedatafiles.read_data_file(fname)

# Unpack
mesh = solver.mesh
physics = solver.physics

# Error
TotErr,_ = post.get_error(mesh, physics, solver, "Density")

# Plot
plot.prepare_plot()
plot.plot_solution(mesh, physics, solver, "Energy", plot_numerical=True, 
			plot_exact=False, plot_IC=False, create_new_figure=True, 
			fmt='bo', legend_label="DG")
plot.plot_solution(mesh, physics, solver, "Energy", plot_exact=True, 
			plot_numerical=False, create_new_figure=False, fmt='k-')
plot.plot_solution(mesh, physics, solver, "Energy", plot_IC=True, 
			plot_numerical=False, create_new_figure=False, fmt='k--')

plot.save_figure(file_name='energy', file_type='pdf', crop_level=2)

plot.show_plot()
