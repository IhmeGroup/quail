This case involves 1D constant advection of a sine wave.
  - Periodic boundaries in the x-direction
  	- Optional `BoundaryConditions` can be uncommented (`PeriodicBoundariesX` would also need to be commented)
  - 3rd-order colocated scheme with Gauss-Lobatto points, i.e. solution nodes and quadrature points are prescribed to be the same
  	- `ColocatedPoints` can be set to `False` if desired
  - Post-processing
  	- Total error printed to display
    - Scalar profile with initial and exact solution displayed
  - Additional Notes:
  	- `restart_ader.py` shows how to restart from a `*.pkl` file. 
  	- Try the following:
  		- Change `AutoPostProcess` to `False` in `constant_advection.py`
  		- Run the following command:
```sh
$ quail constant_advection.py; quail restart_ader.py -p ader_post.py
```