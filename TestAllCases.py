import sys; sys.path.append('./src')
import code
import matplotlib as mpl
from matplotlib import pyplot as plt
from numpy.testing import assert_almost_equal
import argparse


#print(args.accumulate(args.integers))


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'On', 'on'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'Off', 'off'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Options for DG-Python Tests')
parser.add_argument('-ader', type=str2bool, nargs='?',const=True, default=False,
                   help='test ADER cases {Default: Off}')
parser.add_argument('-src', type=str2bool, nargs='?', const=True, default=False,
	               help='test source term cases')
parser.add_argument('-test_1D',type=str2bool, nargs='?',const=True, default=True,
					help='test 1D cases, {Default: On}')
parser.add_argument('-test_2D',type=str2bool, nargs='?',const=True, default=True,
					help='test 2D cases, {Default: On}')

parser.add_argument('-all',type=str2bool, nargs='?',const=True, default=False,
					help='test ALL cases, {Default: Off}')

args = parser.parse_args()

if args.all is True:
	args.ader = True
	args.src = True
'''
Setup
'''

# Make plots non-blocking
plt.ion()

# Desired precision
decimal = 14


'''
1D Test Cases
'''
if args.test_1D is True:
	'''
	1D scalar advection
	'''

	# Linear advection
	sys.path.append('Cases/Scalar_1D/LinearAdvection')
	import LinearAdvection
	assert_almost_equal(LinearAdvection.TotErr, 0.000208876119327, decimal=decimal)
	print('Pass 1D Linear Advection')

	if args.src is True:
		# Linear advection w/source term
		import DampingLinearAdvection
		assert_almost_equal(DampingLinearAdvection.TotErr,0.000372526796142, decimal=decimal)
		print('Pass 1D Damping Linear Advection')

	if args.ader is True:
		# Linear advection ADER
		import ADER_LinearAdvection
		assert_almost_equal(ADER_LinearAdvection.TotErr,0.000163923152029, decimal=decimal)
		print('Pass 1D ADER Linear Advection')

	'''
	1D Euler
	'''

	# Smooth isentropic flow
	sys.path.append('Cases/Euler_1D/SmoothIsentropicFlow')
	import SmoothIsentropicFlow
	assert_almost_equal(SmoothIsentropicFlow.TotErr, 0.000682718239352, decimal=decimal)
	print('Pass 1D Smooth Isentropic Flow [Euler 1D]')

	# Moving shock
	sys.path.append('Cases/Euler_1D/MovingShock')
	import MovingShock
	assert_almost_equal(MovingShock.TotErr, 0.189162974501726, decimal=decimal)
	print('Pass 1D Moving Shock [Euler 1D]')

	if args.ader is True:
		# Smooth isentropic flow ADER
		import ADER_SmoothIsentropicFlow
		assert_almost_equal(ADER_SmoothIsentropicFlow.TotErr, 0.000658156554625, decimal=decimal)
		print('Pass 1D ADER Smooth Isentropic Flow [Euler 1D]')

if args.test_2D is True:

	'''
	2D Euler
	'''

	# Vortex propagation
	sys.path.append('Cases/Euler_2D/VortexPropagation')
	import VortexPropagation
	assert_almost_equal(VortexPropagation.TotErr, 0.009663006404448, decimal=decimal)
	print('Pass 2D Vortex Propagation')


	# Flow over bump
	sys.path.append('Cases/Euler_2D/FlowOverBump')
	import FlowOverBump
	assert_almost_equal(FlowOverBump.TotErr, 0.000012270459601, decimal=decimal)
	print('Pass 2D Flow Over Bump')



'''
Finish
'''

print('Done running all cases')
