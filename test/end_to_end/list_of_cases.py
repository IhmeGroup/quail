# Default tolerances
rtol = 1e-14
atol = 1e-14

# All currently available markers
one_d = 'one_d'
two_d = 'two_d'
dg = 'dg'
ader = 'ader'
splitting = 'splitting'
source = 'source'
diffusion = 'diffusion'

# A dictionary containing the directories of each test case. When adding a new
# test case, the directory needs to be added to this list as well as a list of
# markers for it.
case_dirs = {
	'scalar/1D/constant_advection' : [[one_d, dg], [rtol, atol]],
	'scalar/1D/inviscid_burgers' : [[one_d, dg], [rtol, atol]],
	'scalar/1D/damping_sine_wave/dg' : [[one_d, dg, source], [rtol, atol]],
	'scalar/1D/damping_sine_wave/ader' : [[one_d, ader, source], [rtol, atol]],
	'scalar/1D/damping_sine_wave/splitting' : [[one_d, dg, splitting, source], 
		[rtol, atol]],
	'scalar/2D/constant_advection/dg' : [[two_d, dg], [1e-9, 1e-13]],
	'euler/1D/smooth_isentropic_flow' : [[two_d, dg], [rtol, atol]],
	'euler/2D/flow_over_bump' : [[two_d, dg], [rtol, atol]],
	'euler/2D/isentropic_vortex' : [[two_d, dg], [rtol, atol]],
	'scalar/1D/constant_advection_diffusion/dg' : [[one_d, dg, diffusion],
		[rtol, atol]],
	'scalar/1D/constant_advection_diffusion/ader/interpolate' : \
		[[one_d, ader, diffusion], [rtol, atol]],
	'scalar/1D/constant_advection_diffusion/ader/l2projection' : \
		[[one_d, ader, diffusion], [rtol, atol]],
	'scalar/2D/constant_advection_diffusion/dg' : \
		[[two_d, dg, diffusion], [rtol, atol]],
	'scalar/2D/constant_advection/ader' : [[two_d, ader], [rtol, atol]],
	'navierstokes/2D/manufactured_solution/dg' : [[two_d, ader, diffusion],
		[rtol, atol]],
	'scalar/2D/constant_advection_diffusion/ader' : \
		[[two_d, ader, diffusion], [rtol, atol]],
	'euler/1D/sod_problem/nolimiter' : [[one_d, dg], [rtol ,atol]],
	'euler/1D/sod_problem/wenolimiter' : [[one_d, dg], [200.0, 1e-13]],
	}
