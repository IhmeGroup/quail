import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles
import numpy as np
import cantera as ct

# Read data file
fname = "Data_final.pkl"
solver = readwritedatafiles.read_data_file(fname)

# Unpack
mesh = solver.mesh
physics = solver.physics

fname = "Data_base_p0.pkl"
print(fname)
solver2 = readwritedatafiles.read_data_file(fname)

# Unpack
mesh2 = solver2.mesh
physics2 = solver2.physics




# Flag for num vs avg
num = True
avg = False

''' Plot '''
#########################################################################################

### Density
plot.prepare_plot()
# Exact solution
plot.plot_solution(mesh2, physics2, solver2, "Density", plot_numerical=False, 
		plot_exact=True, plot_IC=False, create_new_figure=True, 
		fmt='k-.', legend_label="Exact")
## Base solution
plot.plot_solution(mesh2, physics2, solver2, "Density", plot_numerical=num, 
 		plot_exact=False, plot_IC=False, plot_average=avg,
 		create_new_figure=False, fmt='bo-', legend_label="Euler")
# DG solution
plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=num, 
 		plot_exact=False, plot_IC=False, plot_average=avg,
 		create_new_figure=False, fmt='ro-', legend_label="Euler+Multispecies")

#########################################################################################

# ### Pressure
# Exact solution
plot.plot_solution(mesh2, physics2, solver2, "Pressure", plot_numerical=False, 
		plot_exact=True, plot_IC=False, create_new_figure=True, 
		fmt='k-.', legend_label="Exact")
# Base solution
plot.plot_solution(mesh2, physics2, solver2, "Pressure", plot_numerical=num, 
 		plot_exact=False, plot_IC=False, plot_average=avg,
 		create_new_figure=False, fmt='bo-', legend_label="Euler")
# DG solution
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=num, 
 		plot_exact=False, plot_IC=False, plot_average=avg,
 		create_new_figure=False, fmt='ro-', legend_label="Euler+Multispecies")
#########################################################################################

## ### Temperature
# Exact solution
plot.plot_solution(mesh2, physics2, solver2, "Temperature", plot_numerical=False, 
		plot_exact=True, plot_IC=False, create_new_figure=True, 
		fmt='k-.', legend_label="Exact")
# Base solution
plot.plot_solution(mesh2, physics2, solver2, "Temperature", plot_numerical=num, 
 		plot_exact=False, plot_IC=False, plot_average=avg,
 		create_new_figure=False, fmt='bo-', legend_label="Euler")
## DG solution
plot.plot_solution(mesh, physics, solver, "Temperature", plot_numerical=num, 
 		plot_exact=False, plot_IC=False, plot_average=avg,
 		create_new_figure=False, fmt='ro-', legend_label="Euler+Multispecies")

#########################################################################################

### Specific Heat Ratio
# Exact solution
plot.plot_solution(mesh2, physics2, solver2, "SpecificHeatRatio", plot_numerical=False, 
		plot_exact=True, plot_IC=False, create_new_figure=True, 
		fmt='k-.', legend_label="Exact")
# Base solution
plot.plot_solution(mesh2, physics2, solver2, "SpecificHeatRatio", plot_numerical=num, 
 		plot_exact=False, plot_IC=False, plot_average=avg,
 		create_new_figure=False, fmt='bo-', legend_label="Euler")
## DG solution
plot.plot_solution(mesh, physics, solver, "SpecificHeatRatio", plot_numerical=num, 
 		plot_exact=False, plot_IC=False, plot_average=avg,
 		create_new_figure=False, fmt='ro-', legend_label="Euler+Multispecies")

#########################################################################################
## Velocity 
# Exact solution
#plot.plot_solution(mesh, physics, solver, "Velocity", plot_numerical=False, 
#		plot_exact=True, plot_IC=False, create_new_figure=True, 
#		fmt='k-.', legend_label="Exact")
## DG solution
#plot.plot_solution(mesh2, physics2, solver2, "Velocity", plot_numerical=num, 
#		plot_exact=False, plot_IC=False, plot_average=avg,
#		create_new_figure=False, fmt='bo-', legend_label="Euler")
#
## DG solution
#plot.plot_solution(mesh, physics, solver, "Velocity", plot_numerical=num, 
#		plot_exact=False, plot_IC=False, plot_average=avg,
#		create_new_figure=False, fmt='ro-', legend_label="Euler+Multispecies")
#

plot.show_plot()
