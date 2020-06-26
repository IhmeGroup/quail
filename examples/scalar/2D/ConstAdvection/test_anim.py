import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DATA = np.random.randn(800).reshape(10,10,8)


fig,ax = plt.subplots()

def animate(i):
	print(i)
	ax.clear()
	ax.contourf(DATA[:,:,i])
	ax.set_title('%03d'%(i)) 

interval = 2#in seconds     
ani = animation.FuncAnimation(fig,animate,5,interval=interval*1e+3,blit=False)

plt.show()