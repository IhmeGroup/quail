import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

### Postprocess
fname = "Data_final.pkl"
solver1 = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver1.Time)

# Unpack
mesh1 = solver1.mesh
physics1 = solver1.EqnSet

### Postprocess
# fname = "Data_Final.pkl"
# solver2 = readwritedatafiles.read_data_file(fname)
# # print('Solution Final Time:', solver.Time)

# # Unpack
# mesh2 = solver2.mesh
# physics2 = solver2.EqnSet

# Error
# TotErr,_ = post.L2_error(mesh, physics, solver, "Density")
# Plot
plot.PreparePlot()
skip=0
plot.plot_solution(mesh1, physics1, solver1, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='b-', legend_label="DG", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip, show_elem_IDs=False)
plot.plot_solution(mesh1, physics1, solver1, "Density", plot_numerical=False, plot_exact=True, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='k--', legend_label="DG", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip)

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
# 			ylabel=None, fmt='k-', legend_label="DG", equidistant_pts=True, 
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
