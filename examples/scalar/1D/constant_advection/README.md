This case involves 1D constant advection of a sine wave.
  - Periodic boundaries in the x-direction
  	- Optional `BoundaryConditions` can be uncommented (`PeriodicBoundariesX` would also need to be commented)
  - 3rd order nodal basis using Gauss Lobatto quadrature with colocated nodes. 
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