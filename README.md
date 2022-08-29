[![Circle CI](https://circleci.com/gh/IhmeGroup/quail.svg?style=shield&circle-token=339239199ddbaf2fd8c1e460931e47b792a13389)](https://circleci.com/gh/IhmeGroup/quail)
[![codecov](https://codecov.io/gh/IhmeGroup/quail/branch/main/graph/badge.svg?token=207AUO9T3V)](https://codecov.io/gh/IhmeGroup/quail)
![GitHub](https://img.shields.io/github/license/IhmeGroup/quail)
<!-- [![GitHub issues](https://img.shields.io/github/issues/IhmeGroup/quail.svg)](https://github.com/IhmeGroup/quail/issues) -->

<p align="center">
  <a href="https://github.com/IhmeGroup/quail"><img alt="quail_logo" src="https://user-images.githubusercontent.com/55554103/99025045-c0ddb780-251c-11eb-9cdb-0bed0269b434.png" width="55%"></a>
</p>

<!-- <p align="center">
  <a href="https://github.com/IhmeGroup/quail"><img alt="quail_logo" src="https://user-images.githubusercontent.com/55554103/144659118-99e88718-694f-412f-9a63-565aacb19cfd.png" width="55%"></a>
</p> -->

Quail is a lightweight, open-source discontinuous Galerkin code written in Python for teaching and prototyping. Currently, Quail solves first-order and second-order nonlinear systems of partial differential equations.

### Setup
Python 3.7 or higher is required. The following libraries should also be installed (tested version numbers provided):
  - NumPy 1.17.4, 1.22.3
  - Matplotlib 3.3.1, 3.5.1
  - SciPy 1.4.1, 1.7.3
  - LaTeX (used for post-processing)

Optional libraries:
  - pytest 6.2.4 (see `test/` directory for usage)
  - cantera 2.5.1 (see `examples/zerodimensional/model_psr`) 

For convenience, the Quail src directory can be added to PATH. The driver script (`quail`) is located in this directory.
```sh
$ export PATH=$PATH:/your/quail/directory/src
```
The above line can also be added to the appropriate file (e.g., `~/.bashrc`, `~/.bash_profile`, `~/.profile`) and sourced.


### Using Quail 
A suite of example 1D and 2D cases for scalar equations, the compressible Euler equations, the compressible Navier-Stokes equations, and chemically reacting flow is available in the `examples` directory. Also showcased are the ADERDG scheme and various stabilization methods (positivity-preserving limiter, WENO limiter, and artificial viscosity). For instance, to run the 2D isentropic vortex case, do the following:
```sh
$ cd examples/euler/2D/isentropic_vortex/
$ quail isentropic_vortex.py
```
Depending on the configuration of your machine, the above command (`quail isentropic_vortex.py`) may not work. In that case, try replacing the first line of `src/quail` with `#!/usr/bin/env python3`. If that still doesn't work, run the following command instead:
```sh
$ python /your/quail/directory/src/quail isentropic_vortex.py
```
Note that this command doesn't require the Quail src directory to be added to PATH.

Additional tools for performing dissipation and dispersion analysis and plotting basis functions are available in the `tools` directory. To perform said analysis, do the following:
```sh
$ cd tools/dissipation_dispersion_analysis/
$ python plot_dissipation_dispersion_relations.py 
```
Settings can be changed directly in `plot_dissipation_dispersion_relations.py`.
To plot 1D basis functions, do the following:
```sh
$ cd tools/plot_basis_functions/
$ python plot_segment_basis_fcn.py  
```
Settings can be changed directly in `plot_segment_basis_fcn.py`. Basis functions for triangles and quadrilaterals can also be plotted.

### Citing Quail

If you publish work which mentions Quail, please cite the following [paper](https://authors.elsevier.com/sd/article/S235271102200005X):

````BibTex
@article{CHING2022100982,
title = {Quail: A lightweight open-source discontinuous Galerkin code in Python for teaching and prototyping},
journal = {SoftwareX},
volume = {17},
pages = {100982},
year = {2022},
issn = {2352-7110},
doi = {https://doi.org/10.1016/j.softx.2022.100982},
url = {https://www.sciencedirect.com/science/article/pii/S235271102200005X},
author = {Eric J. Ching and Brett Bornhoft and Ali Lasemi and Matthias Ihme},
}
````

### Additional information
For those interested in contributing to Quail, please see [`CONTRIBUTING.md`](https://github.com/IhmeGroup/quail/blob/main/CONTRIBUTING.md). Additional details on Quail and the discontinuous Galerkin method can be found in the included documentation ([`docs/documentation.pdf`](https://github.com/IhmeGroup/quail/blob/main/docs/documentation.pdf)). Links to video tutorials are provided as well. Please submit issues and questions on the github page.
