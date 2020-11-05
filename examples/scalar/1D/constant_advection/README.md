This case involves 1D constant advection of a sine wave.
  - Periodic boundaries in the x-direction
  	- Optional BoundaryConditions can be uncommented (PeriodicBoundariesX would also need to be commented)
  - 3rd order nodal basis using Gauss Lobatto quadrature with colocated nodes. 
  	- ColocatedPoints can be set to False
  - Post-processing
  	- Total error printed to display
    - Scalar profile with initial and exact solution displayed