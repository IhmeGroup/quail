This case involves the well-known Sod problem configuration.
The Sod problem is a specific type of Riemann problem.
  - Left and right states are defined as follows:
    - `uL = [1., 0., 1.]` and `uR = [0., .1, .5]`
    (`rho, u, p` respectively)
  - Post-processing plots the exact solution against the
  density, pressure, and velocity.
  - By default, the case uses the positivity preserving limiter to improve
    stability. Instead, artificial viscosity can be used, which has the added
    benefit of smoothing out oscillations near discontinuities. This can be
    turned on by changing `"ArtificialViscosity"` to `True` in the input file.

Shown below are the results using the positivity preserving limiter.
<p align="left">
  <img alt="Density" src="https://user-images.githubusercontent.com/55554103/102403873-84362d80-3fac-11eb-9685-0d585e428238.png" width="33%"></a>
  <img alt="Pressure" src="https://user-images.githubusercontent.com/55554103/102404414-53a2c380-3fad-11eb-990b-3c4fa6283c0d.png" width="33%"></a>
  <img alt="Velocity" src="https://user-images.githubusercontent.com/55554103/102404491-6ae1b100-3fad-11eb-9be0-ffee7c27046a.png" width="33%"></a>
</p>

Shown below are the results using artificial viscosity.
<p align="left">
  <img alt="Density" src="https://user-images.githubusercontent.com/41877612/144340298-e572c82b-2054-4016-ab4e-f104c621c185.png" width="33%"></a>
  <img alt="Pressure" src="https://user-images.githubusercontent.com/41877612/144340302-866a9b9a-a555-447a-9dfa-cee840a67ccd.png" width="33%"></a>
  <img alt="Velocity" src="https://user-images.githubusercontent.com/41877612/144340306-f9ddd5e8-c0e0-436f-9b42-3d421a9552ca.png" width="33%"></a>
</p>
