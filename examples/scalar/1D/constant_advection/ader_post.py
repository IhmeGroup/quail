import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

### Postprocess RestartFile First
fname = "Data_final.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Restart File Start Time:', solver.time)

# Unpack DG Solution
mesh = solver.mesh
physics = solver.physics

# Plot
plot.prepare_plot()
plot.plot_solution(mesh, physics, solver, "Scalar", plot_numerical=True,
		plot_exact=False, plot_IC=False, create_new_figure=True, fmt='go-',
		legend_label="DG RestartFile")

### Postprocess ADER Solution Second
fname = "ader_final.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)

# Unpack ADERDG Solution
mesh = solver.mesh
physics = solver.physics

# Plot
plot.plot_solution(mesh, physics, solver, "Scalar", plot_numerical=True,
		plot_exact=False, plot_IC=False, create_new_figure=False, fmt='bo',
		legend_label="ADERDG")
plot.plot_solution(mesh, physics, solver, "Scalar", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')

plot.show_plot()
