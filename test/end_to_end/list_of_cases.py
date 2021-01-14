# All currently available markers
dg = 'dg'
ader = 'ader'
splitting = 'splitting'
source  = 'source'

# A dictionary containing the directories of each test case. When adding a new test
# case, the directory needs to be added to this list as well as a list of
# markers for it.
case_dirs = {
        'cases/scalar/1D/constant_advection' : [dg],
        'cases/scalar/1D/inviscid_burgers' : [dg],
        'cases/scalar/1D/damping_sine_wave/dg' : [dg, source],
        'cases/scalar/1D/damping_sine_wave/ader' : [ader, source],
        'cases/scalar/1D/damping_sine_wave/splitting' : [splitting, source],
        'cases/scalar/2D/constant_advection' : [dg],
        'cases/euler/1D/moving_shock' : [dg],
        'cases/euler/1D/smooth_isentropic_flow' : [dg],
        'cases/euler/2D/flow_over_bump' : [dg],
        'cases/euler/2D/isentropic_vortex' : [dg],
        }
