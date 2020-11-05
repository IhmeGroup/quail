This case involves subsonic flow over a cosine-shaped bump.
  - Gmsh mesh file with quadratic approximation of geometry
  - Zero entropy generation in exact solution
  - Should run longer for "true" steady-state, but error should not change significantly
  - Polynomial order sequencing
    - p = 0 calculation (`p0.py`)
    - p = 1 calculation restarting from final p = 0 solution (`p1.py`)
    - p = 2 calculation restarting from final p = 1 solution (`p2.py`)
  - Post-processing (on final p = 2 solution)
    - Pressure contour with mesh and element IDs displayed
    - Entropy contour with mesh displayed
    - Pressure in x-direction along wall plotted
      - Boundary integral gives drag force in x-direction

To run the entire simulation (with polynomial order sequencing) in one command, do the following:
```sh
$ quail p0.py; quail p1.py; quail p2.py -p post_process.py
```

