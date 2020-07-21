import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

### Postprocess
fname = "exact.pkl"
solver1 = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver1.Time)
solver1.Time = 0.25
# Unpack
mesh1 = solver1.mesh
physics1 = solver1.physics
# code.interact(local=locals())
# mesh1.Coords += 0.5

### Postprocess
# fname = "Data_Final.pkl"
# solver2 = readwritedatafiles.read_data_file(fname)
# # print('Solution Final Time:', solver.Time)

# # Unpack
# mesh2 = solver2.mesh
# physics2 = solver2.physics

# Error
# TotErr,_ = post.L2_error(mesh, physics, solver, "Density")
# Plot
plot.PreparePlot()
skip=0
plot.plot_solution(mesh1, physics1, solver1, "Density", plot_numerical=False, plot_exact=True, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='k-.', legend_label="Exact", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip, show_elem_IDs=False)
# plot.plot_solution(mesh1, physics1, solver1, "Density", plot_numerical=False, plot_exact=False, plot_IC=True, create_new_figure=False, 
# 			ylabel=None, fmt='k-.', legend_label="DG", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip)

# fname = "nolimiter_cfl0p025.pkl"
# solver2 = readwritedatafiles.read_data_file(fname)
# print('Solution Final Time:', solver2.Time)
# # Unpack
# mesh2 = solver2.mesh
# physics2 = solver2.physics

# plot.plot_solution(mesh2, physics2, solver2, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
# 			ylabel=None, fmt='b-', legend_label="NoLimiter", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False,skip=0, show_elem_IDs=False)

fname = "Data_final.pkl"
solver2 = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver2.Time)
# Unpack
mesh2 = solver2.mesh
physics2 = solver2.physics

plot.plot_solution(mesh2, physics2, solver2, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='g-', legend_label="NoLimiter", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=0, show_elem_IDs=False)


# fname = "nolimiter_cfl0p1.pkl"
# solver2 = readwritedatafiles.read_data_file(fname)
# print('Solution Final Time:', solver2.Time)
# # Unpack
# mesh2 = solver2.mesh
# physics2 = solver2.physics

# plot.plot_solution(mesh2, physics2, solver2, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
# 			ylabel=None, fmt='b-', legend_label="CFL=0.1", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False,skip=0, show_elem_IDs=False)
# plot.plot_solution(mesh1, physics1, solver1, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			# ylabel=None, fmt='bx-', legend_label="DG", equidistant_pts=True, 
			# include_mesh=False, regular_2D=False, equal_AR=False,skip=skip)
# plot.plot_solution(mesh2, physics2, solver2, "Velocity", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
# 			ylabel=None, fmt='go', legend_label="DG", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False, skip=7)


# plot.plot_solution(mesh, physics, solver, "Energy", plot_exact=True, plot_numerical=False, create_new_figure=False, fmt='k-')
# plot.plot_solution(mesh, physics, solver, "Energy", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')
# plot.SaveFigure(FileName='Velocity', FileType='pdf', CropLevel=2)

# plot.plot_solution(mesh1, physics1, solver1, "Density", plot_numerical=False, plot_exact=True, plot_IC=False, create_new_figure=False, 
#			ylabel=None, fmt='k-', legend_label="DG", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip)
#plot.plot_solution(mesh1, physics1, solver1, "Density", plot_numerical=False, plot_exact=True, plot_IC=False, create_new_figure=False, 
#			ylabel=None, fmt='k-', legend_label="DG", equidistant_pts=True, 
#			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip)

# plot.plot_solution(mesh2, physics2, solver2, "Pressure", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
# 			ylabel=None, fmt='go', legend_label="DG", equidistant_pts=True, 
# 			include_mesh=False, regular_2D=False, equal_AR=False, skip=7)
# plot.plot_solution(mesh, physics, solver, "Pressure", plot_exact=True, plot_numerical=False, create_new_figure=False, fmt='k-')
# plot.plot_solution(mesh, physics, solver, "Pressure", plot_IC=True, plot_numerical=False, create_new_figure=False, fmt='k--')
# plot.SaveFigure(FileName='Pressure', FileType='pdf', CropLevel=2)
# plot.PlotSolution(mesh, physics, solver, "Energy", PlotExact=True, PlotIC=True, legend_label="$p=2$")
# plot.PlotSolution(mesh, physics, solver, "Pressure", create_new_figure=False, legend_label="$p=2$")

# plot.SaveFigure(FileName='SmoothIsentropicFlow', FileType='pdf', CropLevel=2)

plot.ShowPlot()
