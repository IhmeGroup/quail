# All currently available markers
one_d = 'one_d'
two_d = 'two_d'
dg = 'dg'
ader = 'ader'
splitting = 'splitting'
source = 'source'

# A dictionary containing the directories of each test case. When adding a new
# test case, the directory needs to be added to this list as well as a list of
# markers for it.
case_dirs = {
	'scalar/1D/constant_advection' : [one_d, dg],
	'scalar/1D/inviscid_burgers' : [one_d, dg],
	'scalar/1D/damping_sine_wave/dg' : [one_d, dg, source],
	'scalar/1D/damping_sine_wave/ader' : [one_d, ader, source],
	'scalar/1D/damping_sine_wave/splitting' : [one_d, dg, splitting, source],
	'scalar/2D/constant_advection' : [two_d, dg],
	'euler/2D/gravity_riemann' : [two_d, dg],
	'euler/1D/smooth_isentropic_flow' : [two_d, dg],
	'euler/2D/flow_over_bump' : [two_d, dg],
	'euler/2D/isentropic_vortex' : [two_d, dg],
	}
