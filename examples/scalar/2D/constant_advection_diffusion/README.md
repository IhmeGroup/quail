This case involves 2D constant advection and diffusion of a Gaussian pulse.
  - Full state boundaries in the x- and y-directions
  - 4th order Lagrange basis on uniform quadrilateral elements (default 8 x 8 mesh)
  - Post-processing
    - Scalar contour with mesh displayed
  - After running a simulation, create an animation with the command below. An MP4 file can be saved if the appropriate tools for processing video files are installed (e.g., FFmpeg).

```sh
$ quail -p create_anim.py
```

We also include additional scripts for checking the grid convergence for arbitrary orders. To check the convergence rates users can run the following:

```sh
$ quail script_convergence.py
```
This will open the `convergence_inputs.py` input deck and will automatically run it for a series of different mesh sizes and orders. Users are then encouraged to run the `process.py` file found in the `convergence_testing` directory. An example graph is shown below.
<p align="center">
  <img alt="anim" src="https://user-images.githubusercontent.com/55554103/137525330-eee8a556-d17e-456e-9d31-b4e48a9d255f.gif" width="48%"></a>
  <img alt="conv" src="https://user-images.githubusercontent.com/55554103/137531264-0b6fde79-6bcd-4882-b9e4-54ab1fc6e133.png" width="46%"></a>
</p>
