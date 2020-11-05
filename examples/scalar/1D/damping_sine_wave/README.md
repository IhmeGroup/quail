This case involves 1D advection of a damping sine wave.
  - Periodic boundaries in the x-direction
  - Post-processing
  	- Total error printed to display
    - Scalar profile with initial and exact solution displayed
  - Additional Notes:
    - The parameter `nu` can force this equation to be stiff if a
    larger value is used.
    - `damping_sine_wave_strang.py` shows how to use operator splitting
    to handle a stiff source term.
    - Try running the input files with `nu = 100.0`. 
  	
