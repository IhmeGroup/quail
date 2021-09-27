This case involves 2D constant advection of a Gaussian pulse.
  - Periodic boundaries in the x- and y-directions
  - 10th order modal basis on uniform triangular elements
  - Post-processing
    - Scalar contour with mesh and element IDs displayed
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
  <img alt="anim" src="https://user-images.githubusercontent.com/10471417/99013103-c7b8fa00-251d-11eb-8405-634e6c9a4c16.gif" width="48%"></a>
  <img alt="conv" src="https://user-images.githubusercontent.com/55554103/134962970-3f044c9a-9cf9-4f86-919c-9b8aca535130.png" width="48%"></a>
</p>
