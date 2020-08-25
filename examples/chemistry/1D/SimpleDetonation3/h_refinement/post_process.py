import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

skip = 0
plot.prepare_plot()

### Postprocess
fname = "150_final.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)
mesh = solver.mesh
physics = solver.physics

plot.plot_solution(mesh, physics, solver, "Temperature", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='-', legend_label="nElem=150", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False, skip=skip)

fname = "300_final.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)
mesh = solver.mesh
physics = solver.physics

plot.plot_solution(mesh, physics, solver, "Temperature", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='-', legend_label="nElem=300", equidistant_pts=True, 
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

plot.save_figure(file_name='CFL_Temp', file_type='pdf', crop_level=2)

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
# plot.save_figure(file_name='Energy', file_type='pdf', crop_level=2)

# plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
# 			ylabel=None, fmt='bo', legend_label="DG", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False)
# plot.plot_solution(mesh, physics, solver, "Pressure", plot_exact=True, plot_numerical=False, create_new_figure=False, fmt='k-')
# plot.plot_solution(mesh, physics, solver, "Pressure", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')
# plot.save_figure(file_name='Pressure', file_type='pdf', crop_level=2)
# plot.PlotSolution(mesh, physics, solver, "Energy", PlotExact=True, PlotIC=True, legend_label="$p=2$")
# plot.PlotSolution(mesh, physics, solver, "Pressure", create_new_figure=False, legend_label="$p=2$")

# plot.save_figure(file_name='SmoothIsentropicFlow', file_type='pdf', crop_level=2)

plot.show_plot()
