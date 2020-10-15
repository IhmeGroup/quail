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

# Error
#TotErr, _ = post.get_error(mesh, physics, solver, "Scalar")
# Plot
plot.prepare_plot()
# plot.PlotSolution(mesh, physics, solver, "Scalar", PlotExact=True, PlotIC=True, Label="u")
plot.plot_solution(mesh, physics, solver, "Scalar", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='bo', legend_label="DG", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False, skip = 5)
#plot.plot_solution(mesh, physics, solver, "Scalar", plot_exact=True, plot_numerical=False, create_new_figure=False, fmt='k-')
plot.plot_solution(mesh, physics, solver, "Scalar", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')


ref_sol = "ref_sol_p0_500elem.pkl"
solver = readwritedatafiles.read_data_file(ref_sol)
mesh = solver.mesh
physics = solver.physics
plot.plot_solution(mesh, physics, solver, "Scalar", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='k-', legend_label="RefSol", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False, skip = 4)
plot.save_figure(file_name='constant_advection', file_type='pdf', crop_level=2)

plot.show_plot()
