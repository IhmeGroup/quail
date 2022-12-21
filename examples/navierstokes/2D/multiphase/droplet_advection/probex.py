import matplotlib.animation as animation
import matplotlib.pyplot as plt
import math

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

plot.prepare_plot(linewidth=0.5)
fig = plt.figure()
ax = plt.gca()

imgs_all = []
j = 0
# Loop through data files
flag=1
for i in range(0,50000):
	print(i)

	# Read data file
	#fname = "./Data_final.pkl"
	fname = "./Data_" + str(i) + ".pkl"
	solver = readwritedatafiles.read_data_file(fname)

	# Unpack
	mesh = solver.mesh
	physics = solver.physics
	''' Plot '''
	### Line probe (y = 1) ###
	# Parameters
	xy2 = [-0.5,0.]; xy1 = [0.5,0.]
#	xy2 = [0.,1.5e-4]; xy1 = [0.007,1.5e-4]
	VSX = True
#	plot.plot_line_probe(mesh, physics, solver, "Density", xy1=xy1, xy2=xy2,
#		plot_numerical=True, plot_exact=False, plot_IC=False,
#		create_new_figure=False, fmt="r-", legend_label="Density", vs_x=VSX)

	# DG solution
	plot.plot_line_probe(mesh, physics, solver, "PhaseField", xy1=xy1, xy2=xy2,
		plot_numerical=True, plot_exact=False, plot_IC=False,
		create_new_figure=True, fmt="y-", legend_label="PhaseField", vs_x=VSX)

	plot.plot_line_probe(mesh, physics, solver, "LevelSet2", xy1=xy1, xy2=xy2,
		plot_numerical=True, plot_exact=False, plot_IC=False,
		create_new_figure=False, fmt="c-", legend_label="LevelSet", vs_x=VSX)

	plot.plot_line_probe(mesh, physics, solver, "YVelocity", xy1=xy1, xy2=xy2,
		plot_numerical=True, plot_exact=False, plot_IC=False,
		create_new_figure=False, fmt="r-", legend_label="YVelocity", vs_x=VSX)
		
	plot.plot_line_probe(mesh, physics, solver, "XVelocity", xy1=xy1, xy2=xy2,
		plot_numerical=True, plot_exact=False, plot_IC=False,
		create_new_figure=False, fmt="b-", legend_label="XVelocity", vs_x=VSX)
		
	plot.plot_line_probe(mesh, physics, solver, "Pressure", xy1=xy1, xy2=xy2,
		plot_numerical=True, plot_exact=False, plot_IC=False,
		create_new_figure=False, fmt="k-", legend_label="Pressure", vs_x=VSX)

	plot.plot_line_probe(mesh, physics, solver, "Density", xy1=xy1, xy2=xy2,
		plot_numerical=True, plot_exact=False, plot_IC=False,
		create_new_figure=False, fmt="g-", legend_label="Density", vs_x=VSX)
		
			

	plt.show()
	
#	name = "./probe/figure" + str(i)
#	plt.savefig(name, dpi=450)
