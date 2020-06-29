import code

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles


plot.PreparePlot(linewidth=0.5)
fig = plt.figure()
ax = plt.gca()


imgs_all = []
j = 0
for i in range(51):
	print(i)
	fname = "Data_" + str(i) + ".pkl"
	solver = readwritedatafiles.read_data_file(fname)
	# Unpack
	mesh = solver.mesh
	physics = solver.EqnSet

	if j == 0:
		ignore_colorbar = False
	else:
		ignore_colorbar = True
	plot.PlotSolution(mesh, physics, solver, "Scalar", create_new_figure=False, Equidistant=True, PlotExact=False, include_mesh=True, 
			Regular2D=True, ShowTriangulation=False, show_elem_IDs=True, ignore_colorbar=ignore_colorbar)

	imgs = ax.collections.copy()

	# plt.show()
	if j == 0:
		imgs_all.append(imgs)
	else:
		nc = len(imgs_all[j-1])
		imgs_all.append(imgs[-nc:])

	j += 1


anim = animation.ArtistAnimation(fig, imgs_all, interval=50, blit=False,
                                repeat_delay=None)

plt.show()

anim.save("anim.mp4")
