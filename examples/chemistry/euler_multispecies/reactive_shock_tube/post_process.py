import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles
import numpy as np
from external.optional_cantera import ct

dgflag = True
aderflag = False

if dgflag == True:
	# Read data file
	fname = "Data_p2.pkl"
	solver = readwritedatafiles.read_data_file(fname)

	# Unpack
	mesh = solver.mesh
	physics = solver.physics

fname = "Data_ader_p2.pkl"
solver3 = readwritedatafiles.read_data_file(fname)

# Unpack
mesh3 = solver3.mesh
physics3 = solver3.physics

name2 = "DG MultiSp"
name3 = "ADERDG MultiSp"

# Flag for num vs avg
num = True
avg = False

''' Plot '''
#########################################################################################

### Density
plot.prepare_plot()
## Exact solution
#plot.plot_solution(mesh2, physics2, solver2, "Density", plot_numerical=False, 
#		plot_exact=True, plot_IC=False, create_new_figure=True, 
#		fmt='k-.', legend_label="Exact")
# DG solution
plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=num, 
 		plot_exact=False, plot_IC=False, plot_average=avg,
 		create_new_figure=True, fmt='ro-', legend_label=name2)

# ADERDG solution
plot.plot_solution(mesh3, physics3, solver3, "Density", plot_numerical=num, 
 		plot_exact=False, plot_IC=False, plot_average=avg,
 		create_new_figure=False, fmt='g-', legend_label=name3)

#plot.save_figure(file_name='density', file_type='pdf', crop_level=2)

#########################################################################################

# ### Pressure
# Exact solution
#plot.plot_solution(mesh2, physics2, solver2, "Pressure", plot_numerical=False, 
#		plot_exact=True, plot_IC=False, create_new_figure=True, 
#		fmt='k-.', legend_label="Exact")
# DG solution
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=num, 
 		plot_exact=False, plot_IC=False, plot_average=avg,
 		create_new_figure=True, fmt='ro-', legend_label=name2)

# ADERDG solution
plot.plot_solution(mesh3, physics3, solver3, "Pressure", plot_numerical=num, 
 		plot_exact=False, plot_IC=False, plot_average=avg,
 		create_new_figure=False, fmt='g-', legend_label=name3)


#plot.save_figure(file_name='pressure', file_type='pdf', crop_level=2)

#########################################################################################

## ### Temperature
## Exact solution
#plot.plot_solution(mesh2, physics2, solver2, "Temperature", plot_numerical=False, 
#		plot_exact=True, plot_IC=False, create_new_figure=True, 
#		fmt='k-.', legend_label="Exact")
## DG solution
plot.plot_solution(mesh, physics, solver, "Temperature", plot_numerical=num, 
 		plot_exact=False, plot_IC=False, plot_average=avg,
 		create_new_figure=True, fmt='ro-', legend_label=name2)

# ADERDG solution
plot.plot_solution(mesh3, physics3, solver3, "Temperature", plot_numerical=num, 
 		plot_exact=False, plot_IC=False, plot_average=avg,
 		create_new_figure=False, fmt='g-', legend_label=name3)

#plot.save_figure(file_name='temperature', file_type='pdf', crop_level=2)

#########################################################################################

### Specific Heat Ratio
# Exact solution
#plot.plot_solution(mesh2, physics2, solver2, "SpecificHeatRatio", plot_numerical=False, 
#		plot_exact=True, plot_IC=False, create_new_figure=True, 
#		fmt='k-.', legend_label="Exact")
# DG solution
plot.plot_solution(mesh, physics, solver, "SpecificHeatRatio", plot_numerical=num, 
 		plot_exact=False, plot_IC=False, plot_average=avg,
 		create_new_figure=True, fmt='ro-', legend_label=name2)

# ADERDG solution
plot.plot_solution(mesh3, physics3, solver3, "SpecificHeatRatio", plot_numerical=num, 
 		plot_exact=False, plot_IC=False, plot_average=avg,
 		create_new_figure=False, fmt='g-', legend_label=name3)

#plot.save_figure(file_name='gamma', file_type='pdf', crop_level=2)

plot.show_plot()
