This case solves the a second order ODE that models a pendulum system. The equation is as follows:

-![CodeCogsEqn](https://user-images.githubusercontent.com/55554103/127205491-4ef2632e-06f8-4b18-860c-68eb89735629.png)
Due to the linear nature of the problem it can be used to test the order of accuracy of a time integration scheme. To test the time integration scheme alone, users can set "TimeStepper" to "ODEIntegrator". In addition they can select "ODEScheme" to be the scheme they would like to test. To conduct a time integration order of accuracy test one can follow these steps.

To create the figure below you follow these steps:
- `cd convergence testing` and 	`mkdir RK4`, `mkdir BDF1`, and `mkdir Trapezoidal`. Then `cd ..`
- Run `quail script.py` (in this directory) to generate the solution files for this case using different time step sizes.
- Do this three times but change the `scheme_name` in `script.py` to `BDF1`, `Trapezoidal`, and `RK4`
	- This generates a set of `*.pkl` files in the `convergence_testing` directory. 
- Next, `cd convergence testing` and run `python process.py`. This should generate the appropriate convergence plots and output the order of convergence between each time step size.
