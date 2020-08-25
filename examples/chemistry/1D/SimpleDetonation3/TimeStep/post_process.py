import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

skip = 0
plot.prepare_plot()

### Postprocess
fname = "Data_0p25.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)
mesh = solver.mesh
physics = solver.physics

plot.plot_solution(mesh, physics, solver, "Temperature", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='-', legend_label="CFL=0.25", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False, skip=skip)

fname = "Data_0p5.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)
mesh = solver.mesh
physics = solver.physics

plot.plot_solution(mesh, physics, solver, "Temperature", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='-', legend_label="CFL=0.5", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False, skip=skip)

# fname = "Da100.pkl"
# solver = readwritedatafiles.read_data_file(fname)
# print('Solution Final Time:', solver.time)
# mesh = solver.mesh
# physics = solver.physics

# plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
# 			ylabel=None, fmt='-', legend_label="Da=100", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False, skip=skip)

# fname = "Da1000.pkl"
# solver = readwritedatafiles.read_data_file(fname)
# print('Solution Final Time:', solver.time)
# mesh = solver.mesh
# physics = solver.physics

# plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
# 			ylabel=None, fmt='-', legend_label="Da=1000", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False, skip=skip)

plot.plot_solution(mesh, physics, solver, "Temperature", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')

plot.save_figure(FileName='CFL_Temp', FileType='pdf', CropLevel=2)

# plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
# 			ylabel=None, fmt='b-', legend_label="DG", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False, skip=skip)
# plot.plot_solution(mesh, physics, solver, "Density", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')
# plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
# 			ylabel=None, fmt='b-', legend_label="DG", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False, skip=skip)
# plot.plot_solution(mesh, physics, solver, "Pressure", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')
# plot.plot_solution(mesh, physics, solver, "Temperature", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
# 			ylabel=None, fmt='b-', legend_label="DG", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False, skip=skip)
# plot.plot_solution(mesh, physics, solver, "Temperature", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')
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
