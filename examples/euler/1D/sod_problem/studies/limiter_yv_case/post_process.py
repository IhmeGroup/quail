import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

### Postprocess
plot.prepare_plot()
skip=0

fname = "case_0.pkl"
solver1 = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver1.time)
solver1.time = 0.25
mesh1 = solver1.mesh
physics1 = solver1.physics


fname = "case_3.pkl"
solver2 = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver2.time)
mesh2 = solver2.mesh
physics2 = solver2.physics


# Density
plot.plot_solution(mesh1, physics1, solver1, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='g-', legend_label="No Limiter", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip, show_elem_IDs=True)
plot.plot_solution(mesh2, physics2, solver2, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='b-', legend_label="PPLimiter", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip, show_elem_IDs=False)

plot.save_figure(FileName='density', FileType='pdf', CropLevel=2)

# Velocity 
plot.plot_solution(mesh1, physics1, solver1, "Velocity", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='g-', legend_label="No Limiter", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip, show_elem_IDs=True)
plot.plot_solution(mesh2, physics2, solver2, "Velocity", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='b-', legend_label="PPLimiter", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip, show_elem_IDs=False)

plot.save_figure(FileName='velocity', FileType='pdf', CropLevel=2)

# Velocity 
plot.plot_solution(mesh1, physics1, solver1, "Pressure", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='g-', legend_label="No Limiter", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip, show_elem_IDs=True)
plot.plot_solution(mesh2, physics2, solver2, "Pressure", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='b-', legend_label="PPLimiter", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip, show_elem_IDs=False)

plot.save_figure(FileName='pressure', FileType='pdf', CropLevel=2)
plot.show_plot()
