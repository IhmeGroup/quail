import matplotlib.animation as animation
import matplotlib.pyplot as plt

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

import numpy as np

nimages = 17 

plot.prepare_plot(linewidth=0.5)
fig = plt.figure()
ax = plt.gca()

imgs_all = []
j = 0
# Loop through data files
for i in range(nimages):
	print(i)

	# Read data file
	fname = "Data_" + str(i) + ".pkl"
	solver = readwritedatafiles.read_data_file(fname)

	# Unpack
	mesh = solver.mesh
	physics = solver.physics

	# Plot solution
	if j == 0:
		ignore_colorbar = False
	else:
		ignore_colorbar = True
	levels = np.array([0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5])
	plot.plot_solution(mesh, physics, solver, "Density", plot_numerical=True, create_new_figure=False, 
			include_mesh=False, regular_2D=True, equal_AR=False, show_elem_IDs=True, ignore_colorbar=ignore_colorbar,
			levels=levels)

	imgs = ax.collections.copy()

	# Add to imgs_all
	if j == 0:
		imgs_all.append(imgs)
	else:
		nc = len(imgs_all[j-1])
		imgs_all.append(imgs[-nc:])

	j += 1

anim = animation.ArtistAnimation(fig, imgs_all, interval=200, blit=False,
		repeat_delay=None)

plt.show()

# Save mp4
anim.save("anim.mp4")
