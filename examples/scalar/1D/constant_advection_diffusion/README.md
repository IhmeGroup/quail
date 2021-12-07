This case involves 1D constant advection and diffusion of a Gaussian pulse.
  - Periodic boundaries in the x-direction
  - 4th order Lagrange basis on uniform segment elements (default 64 element mesh)
  - Post-processing
    - Scalar line plot comparing the exact solution.
  - After running a simulation, create an animation with the command below. An MP4 file can be saved if the appropriate tools for processing video files are installed (e.g., FFmpeg).

```sh
$ quail -p create_anim.py
```

<p align="center">
  <img alt="anim" src="https://user-images.githubusercontent.com/55554103/144388632-b70b7f52-3cc7-4726-bc02-6c7cb7d5a131.gif" width="50%"></a>
</p>
