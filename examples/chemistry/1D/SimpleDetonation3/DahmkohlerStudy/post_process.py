import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

skip = 0
plot.prepare_plot()

### Postprocess
# fname = "Da0.pkl"
# solver = readwritedatafiles.read_data_file(fname)
# print('Solution Final Time:', solver.time)
# mesh = solver.mesh
# physics = solver.physics

# plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
# 			ylabel=None, fmt='-', legend_label="Da=0", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False, skip=skip)

# fname = "Da10.pkl"
# solver = readwritedatafiles.read_data_file(fname)
# print('Solution Final Time:', solver.time)
# mesh = solver.mesh
# physics = solver.physics

# plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
# 			ylabel=None, fmt='-', legend_label="Da=10", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False, skip=skip)

fname = "Da1000.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)
mesh = solver.mesh
physics = solver.physics

plot.plot_solution(mesh, physics, solver, "Temperature", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='-', legend_label="nElem=150", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False, skip=skip)

plot.plot_solution(mesh, physics, solver, "Temperature", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')



fname = "Test_1000_fine.pkl"
solver = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver.time)
mesh = solver.mesh
physics = solver.physics

plot.plot_solution(mesh, physics, solver, "Temperature", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='-', legend_label="nElem=300", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False, skip=skip)

plot.save_figure(FileName='Refine_Temperature_Da1000', FileType='pdf', CropLevel=2)

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
