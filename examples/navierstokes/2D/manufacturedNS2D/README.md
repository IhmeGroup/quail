This case involves a 2D manufactured solution for the Navier-Stokes equations
  - Case is run using ADERDG
  - Order set to P3
  - Post-processing
    - Density, Pressure, x/y-velocity contours 

The manufactured source term is generated using `manufactured_NS_source.py` which uses `sympy` to generate the appropriate sources. These are then added to the  `navierstokes/functions.py` file (already completed for this case). 

Case follows this reference:

Navah, F., and Nadarajah, S. (2018). A comprehensive high-order solver verification methodology for free fluid flows. Aerospace Science and Technology, 80, 101-126. https://doi.org/10.1016/j.ast.2018.07.006


<p align="center">
  <img alt="density" src="https://github.com/IhmeGroup/quail_dev/files/7355365/Density.pdf" width="48%"></a>
  <img alt="pressure" src="https://github.com/IhmeGroup/quail_dev/files/7355367/Pressure.pdf" width="48%"></a>
</p>

<p align="center">
  <img alt="x-velocity" src="https://github.com/IhmeGroup/quail_dev/files/7355370/XVelocity.pdf" width="48%"></a>
  <img alt="y-velocity" src="https://github.com/IhmeGroup/quail_dev/files/7355371/YVelocity.pdf" width="48%"></a>
</p>