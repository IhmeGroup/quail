import matplotlib.animation as animation
import matplotlib.pyplot as plt

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles
import numpy as np


plot.prepare_plot(linewidth=0.5)
fig = plt.figure()
ax = plt.gca()
field = "PhaseField"
#field = "Pressure"
#field = "Velocity"
imgs_all = []
j = 0
# Loop through data files
for i in range(0,610):

	# Read data file
	#fname = "./Data_final.pkl"
	fname = "./Data_" + str(i) + ".pkl"
	solver = readwritedatafiles.read_data_file(fname)

	t = solver.time
	print(i, " time: ", t)
	# Unpack
	mesh = solver.mesh
	physics = solver.physics

	# Plot solution
	if j == 0:
		ignore_colorbar = False
		include_mesh = True
	else:
		ignore_colorbar = True
		include_mesh = False
		
	ignore_colorbar = False
	include_mesh = True
	
	# Plotting the desidered Field
	plot.plot_solution(mesh, physics, solver, field, plot_numerical=True, create_new_figure=False,
			include_mesh=include_mesh, regular_2D=True, equal_AR=True, show_elem_IDs=False,equidistant_pts=False, ignore_colorbar=ignore_colorbar, Interface=False)
			
	# Plotting the interface only
	plot.plot_solution(mesh, physics, solver, "PhaseField", plot_numerical=True,
		create_new_figure=False, include_mesh=False, regular_2D=True,
		equal_AR=True, show_elem_IDs=False, ignore_colorbar=True,
		levels = [-1.0, 0.5, 2.0], clr = "k", linewidths=[0.5,2.0,0.5],linestyles="-",
		Interface=True)
	plt.show()

	imgs = ax.collections.copy()

	# Add to imgs_all
	if j == 0:
		imgs_all.append(imgs)
	else:
		nc = len(imgs_all[j-1])
		imgs_all.append(imgs[-nc:])

	j += 1


anim = animation.ArtistAnimation(fig, imgs_all, interval=50, blit=False,
		repeat_delay=None)

plt.show()

# Save mp4
anim.save("anim.mp4")
