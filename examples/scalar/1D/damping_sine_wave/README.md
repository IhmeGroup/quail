This case involves 1D advection of a damping sine wave.
  - Periodic boundaries in the x-direction
  - Post-processing
  	- Total error printed to display
    - Scalar profile with initial and exact solution displayed
  - Additional Notes:
    - The parameter `nu` can force this equation to be stiff if a
    larger value is used.
    - `damping_sine_wave_splitting.py` shows how to use operator splitting
    to handle a stiff source term.
    - Try running both inputfiles with `nu = -1000.` Which one remains 
    stable?
    - You can also change `TimeStepper` from `Strang` to `Simpler` to 
    try out different splitting schemes.
  - ADERDG: 
    - Lastly, we can use this case to test out the ADERDG scheme. 
    This scheme can also be appropriate for stiff systems. 
    - Try running `damping_sine_wave_ader.py` with `nu = -3.` and
    then with `nu = -1000.`. Was `nu = -1000` unstable?
    - Now try again, but change `SourceTreatmentADER` to be `Implicit`.
    Does the case run stably now?
  	
