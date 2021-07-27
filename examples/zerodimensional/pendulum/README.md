This case solves the a second order ODE that models a pendulum system. The equation is as follows:

-![CodeCogsEqn](https://user-images.githubusercontent.com/55554103/127205002-31f7b001-78b4-4b8d-a075-835da180a84c.gif)
Due to the linear nature of the problem it can be used to test the order of accuracy of a time integration scheme. To test the time integration scheme alone, users can set "TimeStepper" to "ODEIntegrator". In addition they can select "ODEScheme" to be the scheme they would like to test. To conduct a time integration order of accuracy test one can follow these steps.

To create the figure below you follow these steps:
- `cd convergence testing` and 	`mkdir RK4`, `mkdir BDF1`, and `mkdir Trapezoidal`. Then `cd ..`
- Run `quail script.py` (in this directory) to generate the solution files for this case using different time step sizes.
- Do this three times but change the `scheme_name` in `script.py` to `BDF1`, `Trapezoidal`, and `RK4`
	- This generates a set of `*.pkl` files in the `convergence_testing` directory. 
- Next, `cd convergence testing` and run `python process.py`. This should generate the appropriate convergence plots and output the order of convergence between each time step size.

<p align="center">
  <a href="https://github.com/ericjching/DG_Python"><img alt="quail_logo" src="https://user-images.githubusercontent.com/55554103/127051351-544e6c4d-e7a6-4689-bb99-cf8e780cb340.png" width="55%"></a>
</p>
