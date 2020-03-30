import sys; sys.path.append('./src')
import code
import matplotlib as mpl
from matplotlib import pyplot as plt
from numpy.testing import assert_almost_equal



'''
Setup
'''

# Make plots non-blocking
plt.ion()

# Desired precision
decimal = 14



'''
1D scalar advection
'''

# Linear advection
sys.path.append('Cases/Scalar_1D/LinearAdvection')
import LinearAdvection
assert_almost_equal(LinearAdvection.TotErr, 0.000208876119327, decimal=decimal)



'''
1D Euler
'''

# Smooth isentropic flow
sys.path.append('Cases/Euler_1D/SmoothIsentropicFlow')
import SmoothIsentropicFlow
assert_almost_equal(SmoothIsentropicFlow.TotErr, 0.000682718239352, decimal=decimal)

# Moving shock
sys.path.append('Cases/Euler_1D/MovingShock')
import MovingShock
assert_almost_equal(MovingShock.TotErr, 0.172768793744101, decimal=decimal)



'''
2D Euler
'''

# Vortex propagation
sys.path.append('Cases/Euler_2D/VortexPropagation')
import VortexPropagation
assert_almost_equal(VortexPropagation.TotErr, 0.009663006404448, decimal=decimal)

# Flow over bump
sys.path.append('Cases/Euler_2D/FlowOverBump')
import FlowOverBump
assert_almost_equal(FlowOverBump.TotErr, 0.000012270459601, decimal=decimal)



'''
Finish
'''

print('Done running all cases')