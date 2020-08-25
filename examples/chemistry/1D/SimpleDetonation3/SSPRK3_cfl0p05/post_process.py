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

skip = 0
# Error
# TotErr,_ = post.get_error(mesh, physics, solver, "Density")
# Plot
plot.prepare_plot()
plot.plot_solution(mesh, physics, solver, "MassFraction", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='b-', legend_label="DG", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False, skip=skip)
plot.plot_solution(mesh, physics, solver, "MassFraction", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')
plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='b-', legend_label="DG", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False, skip=skip)
plot.plot_solution(mesh, physics, solver, "Density", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='b-', legend_label="DG", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False, skip=skip)
plot.plot_solution(mesh, physics, solver, "Pressure", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')
plot.plot_solution(mesh, physics, solver, "Temperature", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='b-', legend_label="DG", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False, skip=skip)
plot.plot_solution(mesh, physics, solver, "Temperature", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')
# plot.save_figure(FileName='Energy', FileType='pdf', CropLevel=2)

# plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
# 			ylabel=None, fmt='bo', legend_label="DG", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False)
# plot.plot_solution(mesh, physics, solver, "Pressure", plot_exact=True, plot_numerical=False, create_new_figure=False, fmt='k-')
# plot.plot_solution(mesh, physics, solver, "Pressure", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')
# plot.save_figure(FileName='Pressure', FileType='pdf', CropLevel=2)
# plot.PlotSolution(mesh, physics, solver, "Energy", PlotExact=True, PlotIC=True, legend_label="$p=2$")
# plot.PlotSolution(mesh, physics, solver, "Pressure", create_new_figure=False, legend_label="$p=2$")

# plot.save_figure(FileName='SmoothIsentropicFlow', FileType='pdf', CropLevel=2)

plot.show_plot()
