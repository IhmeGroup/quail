This case involves a 2D manufactured solution for the Navier-Stokes equations
  - Case is run using DG
  - Order set to P3
  - Post-processing
    - Density, Pressure, x/y-velocity contours 

The manufactured source term is generated using `manufactured_NS_source.py` which uses `sympy` to generate the appropriate sources. These are then added to the  `navierstokes/functions.py` file (already completed for this case). 

Case follows this reference:

Navah, F., and Nadarajah, S. (2018). A comprehensive high-order solver verification methodology for free fluid flows. Aerospace Science and Technology, 80, 101-126. https://doi.org/10.1016/j.ast.2018.07.006


<p align="center">
  <img alt="density" src="https://user-images.githubusercontent.com/55554103/137536697-662c10aa-2a10-45f8-bc53-34a3e698893e.png" width="48%"></a>
  <img alt="pressure" src="https://user-images.githubusercontent.com/55554103/137536743-9636df51-e629-4a07-956f-bf81ef8f86e9.png" width="46%"></a>
</p>

<p align="center">
  <img alt="x-velocity" src="https://user-images.githubusercontent.com/55554103/137536782-a812af78-7b17-4583-b4ea-8f0ff47bc535.png" width="48%"></a>
  <img alt="y-velocity" src="https://user-images.githubusercontent.com/55554103/137536826-51c038c2-1ced-4399-aaa2-9acaeb752db4.png" width="48%"></a>
</p>
