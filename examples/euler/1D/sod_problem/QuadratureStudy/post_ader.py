import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

### Postprocess
fname = "Data_ADER_GL.pkl"
solver1 = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver1.time)

# Unpack
mesh1 = solver1.mesh
physics1 = solver1.physics

fname = "Data_Exact_Sod.pkl"
exact = readwritedatafiles.read_data_file(fname)
mesh_ex = exact.mesh
physics_ex = exact.physics
### Postprocess
fname = "Data_ADER_GLL.pkl"
solver2 = readwritedatafiles.read_data_file(fname)
# print('Solution Final Time:', solver.time)
# # Unpack
mesh2 = solver2.mesh
physics2 = solver2.physics

fname = "Data_ADER_GLL_ForcedNodes.pkl"
solver3 = readwritedatafiles.read_data_file(fname)

mesh3 = solver3.mesh
physics3 = solver3.physics
# Error

# Plot
plot.prepare_plot()
skip=7

plot.plot_solution(mesh_ex, physics_ex, exact, "Velocity", plot_numerical=False, plot_exact=True, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='k-', legend_label="Exact", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False)
plot.plot_solution(mesh1, physics1, solver1, "Velocity", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='b+', legend_label="GL", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip)

plot.plot_solution(mesh2, physics2, solver2, "Velocity", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='g1', legend_label="GLL", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip)
plot.plot_solution(mesh3, physics3, solver3, "Velocity", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='kx', legend_label="GLL EqualNodes", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip)
plot.save_figure(file_name='Quad_ADER_Velocity', file_type='pdf', crop_level=2)
# plot.save_figure(file_name='GL_only_density', file_type='pdf', crop_level=2)

# plot.plot_solution(mesh1, physics1, solver1, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
# 			ylabel=None, fmt='bx-', legend_label="DG", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip)
# plot.plot_solution(mesh2, physics2, solver2, "Velocity", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
# 			ylabel=None, fmt='go', legend_label="DG", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False, skip=7)


# plot.plot_solution(mesh, physics, solver, "Energy", plot_exact=True, plot_numerical=False, create_new_figure=False, fmt='k-')
# plot.plot_solution(mesh, physics, solver, "Energy", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')
# plot.save_figure(file_name='Velocity', file_type='pdf', crop_level=2)

# plot.plot_solution(mesh1, physics1, solver1, "Pressure", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
# 			ylabel=None, fmt='bx', legend_label="DG", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip)

# plot.plot_solution(mesh2, physics2, solver2, "Pressure", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
# 			ylabel=None, fmt='go', legend_label="DG", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False, skip=7)
# plot.plot_solution(mesh, physics, solver, "Pressure", plot_exact=True, plot_numerical=False, create_new_figure=False, fmt='k-')
# plot.plot_solution(mesh, physics, solver, "Pressure", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')
# plot.save_figure(file_name='Pressure', file_type='pdf', crop_level=2)
# plot.PlotSolution(mesh, physics, solver, "Energy", PlotExact=True, PlotIC=True, legend_label="$p=2$")
# plot.PlotSolution(mesh, physics, solver, "Pressure", create_new_figure=False, legend_label="$p=2$")

# plot.save_figure(file_name='SmoothIsentropicFlow', file_type='pdf', crop_level=2)

plot.show_plot()
