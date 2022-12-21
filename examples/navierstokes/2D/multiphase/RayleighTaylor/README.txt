Rayleigh-Taylor instability at Re=3000 and At = 0.5. 

Test case for validation of gravitational forces without surface tension. 

To run test case:

../quail/src/quail RT.py

For post-processing run:

../quail/src/quail create_anim.py

create_anim.py plots a given field in the domain. You can either keep the plt.show() on to see very time step, or commenting it, you can create an animation of the given field.

The file probex.py plots all the most relevant fields along the x direction. 


References: 

A high order flux reconstruction interface capturing method with a phase field preconditioning procedure, Al-Salami et al. JCP 2021