The one dimensional overdriven detonation case is a temporally evolving unsteady wave. The chemical kinetics are governed by a single-step irreversible reaction where the reaction rate is described in the
Arrhenius form.  
  - Post-processing
    - Pressure, mass fraction, density, and energy profiles.
    - `create_anim.py` can be used to generate an mp4 of a specified variable. 
  - Additional Notes:
    - The initial conditions are determined from the Zeldovich, Neumann, and Doering (ZND) profiles by specifying an overdrive factor of 1.6. Details of solving for the ZND initial condition can be found in Fickett and Davis:

			[1] Fickett, W., and Davis, W., Detonation - Theory and Experiment, Dover Publications, 2000.
    - This case uses the WENO limiter to provide stabilization.

<p align="center">
  <img alt="pressure.mp4" src="https://user-images.githubusercontent.com/55554103/144385206-888db2c1-5ba9-4c0b-86c3-66f646ad217b.gif" width="48%"></a>
  <img alt="massfraction.mp4" src="https://user-images.githubusercontent.com/55554103/144385401-38bec4e5-b672-49c3-b8b6-827f43c92d89.gif" width="48%"></a>

</p>
