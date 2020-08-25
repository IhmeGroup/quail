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
# TotErr,_ = post.get_error(mesh, physics, solver, "ensity")
# Plot
plot.prepare_plot()
plot.plot_solution(mesh, physics, solver, "Energy", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='bo', legend_label="DG", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False)
plot.plot_solution(mesh, physics, solver, "Energy", plot_numerical=False, plot_exact=False, plot_IC=True, create_new_figure=False, 
			ylabel=None, fmt='k--', legend_label="DG", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False)
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='bo', legend_label="DG", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False)
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=False, plot_exact=False, plot_IC=True, create_new_figure=False, 
			ylabel=None, fmt='k--', legend_label="Init", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False)
# plot.PlotSolution(mesh, physics, solver, "Energy", PlotExact=False, PlotIC = True)
# plot.PlotSolution(mesh, physics, solver, "XMomentum", PlotExact=False, PlotIC = True)
# plot.PlotSolution(mesh, physics, solver, "Density", PlotExact=False, PlotIC = True)


# plot.save_figure(FileName='StiffFriction', FileType='pdf', CropLevel=2)

plot.show_plot()
