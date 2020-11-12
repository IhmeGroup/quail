This case involves steady subsonic flow over a cosine-shaped bump.
  - Gmsh mesh file with quadratic approximation of geometry
  - Zero entropy generation in exact solution
  - Should run longer for "true" steady-state, but error should not change significantly for prescribed runtime
  - Can use `bump.geo` to create a finer mesh to obtain better results
    - `Reverse Surface` command needed to maintain a positive Jacobian determinant, i.e. normals should point in the +z-direction
    - Remember to set order 2
    - Will likely need to run for longer
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

<p align="center">
  <img alt="pressure" src="https://user-images.githubusercontent.com/10471417/99009541-2e3a1a00-2516-11eb-8798-99227c40988b.png" width="50%"></a>
</p>
