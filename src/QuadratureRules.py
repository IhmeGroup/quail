import numpy as np


'''
Dictionaries: QuadLinePoints, QuadLineWeights
-------------------
These dictionaries store the Gauss-Legendre quadrature points and weights for the reference line segment

USAGE:
    QuadLinePoints[Order] = quadrature points for Gauss-Legendre quadrature of order Order
    QuadLineWeights[Order] = quadrature weights for Gauss-Legendre quadrature of order Order
'''
QuadLinePoints = {}
QuadLineWeights = {}
for key in [0,1]:
	# Order 1
	QuadLinePoints[key] = np.array([0.5])
	QuadLineWeights[key] = np.array([1.0])
for key in [2,3]:
	# Order 3
	QuadLinePoints[key] = np.array([0.211324865405187, 0.788675134594813])
	QuadLineWeights[key] = np.array([0.500000000000000, 0.500000000000000])
for key in [4,5]:
	# Order 5
	QuadLinePoints[key] = np.array([0.112701665379258, 0.500000000000000, 0.887298334620742])
	QuadLineWeights[key] = np.array([0.277777777777778, 0.444444444444444, 0.277777777777778])
for key in [6,7]:
	# Order 7
	QuadLinePoints[key] = np.array([0.069431844202974, 0.330009478207572, 0.669990521792428, 0.930568155797026])
	QuadLineWeights[key] = np.array([0.173927422568727, 0.326072577431273, 0.326072577431273, 0.173927422568727])
for key in [8,9]:
	# Order 9
	QuadLinePoints[key] = np.array([0.046910077030668, 0.230765344947158, 0.500000000000000, 0.769234655052841,
    0.953089922969332])
	QuadLineWeights[key] = np.array([0.118463442528095, 0.239314335249683, 0.284444444444444, 0.239314335249683,
    0.118463442528095])
for key in [10,11]:
	# Order 11
	QuadLinePoints[key] = np.array([0.033765242898424, 0.169395306766868, 0.380690406958402, 0.619309593041598,
    0.830604693233132, 0.966234757101576])
	QuadLineWeights[key] = np.array([0.085662246189585, 0.180380786524069, 0.233956967286345, 0.233956967286345,
    0.180380786524069, 0.085662246189585])
for key in [12,13]:
	# Order 13
	QuadLinePoints[key] = np.array([0.025446043828621, 0.129234407200303, 0.297077424311301, 0.500000000000000,
    0.702922575688699, 0.870765592799697, 0.974553956171379])
	QuadLineWeights[key] = np.array([0.064742483084435, 0.139852695744638, 0.190915025252560, 0.208979591836735,
    0.190915025252560, 0.139852695744638, 0.064742483084435])
for key in [14,15]:
	# Order 15
	QuadLinePoints[key] = np.array([0.019855071751232, 0.101666761293187, 0.237233795041836, 0.408282678752175,
    0.591717321247825, 0.762766204958164, 0.898333238706813, 0.980144928248768])
	QuadLineWeights[key] = np.array([0.050614268145188, 0.111190517226687, 0.156853322938944, 0.181341891689181,
    0.181341891689181, 0.156853322938944, 0.111190517226687, 0.050614268145188])
for key in [16,17]:
	# Order 17
	QuadLinePoints[key] = np.array([0.015919880246187, 0.081984446336682, 0.193314283649705, 0.337873288298096,
    0.500000000000000, 0.662126711701905, 0.806685716350295, 0.918015553663318, 
    0.984080119753813])
	QuadLineWeights[key] = np.array([0.040637194180787, 0.090324080347429, 0.130305348201468, 0.156173538520001,
    0.165119677500630, 0.156173538520001, 0.130305348201468, 0.090324080347429, 
    0.040637194180787])
for key in [18,19]:
	# Order 19
	QuadLinePoints[key] = np.array([0.013046735741414, 0.067468316655508, 0.160295215850488, 0.283302302935376,
    0.425562830509184, 0.574437169490816, 0.716697697064624, 0.839704784149512,
    0.932531683344492, 0.986953264258586])
	QuadLineWeights[key] = np.array([0.033335672154344, 0.074725674575290, 0.109543181257991, 0.134633359654998,
    0.147762112357376, 0.147762112357376, 0.134633359654998, 0.109543181257991,
    0.074725674575290, 0.033335672154344])
# reshape
for key in range(len(QuadLinePoints)):
	QuadLinePoints[key].shape = -1,1
	QuadLineWeights[key].shape = -1,1


'''
Dictionaries: QuadLinePoints, QuadLineWeights
-------------------
These dictionaries store the Gauss-Legendre quadrature points and weights for the reference quadrilateral

USAGE:
    QuadQuadrilateralPoints[Order] = quadrature points for Gauss-Legendre quadrature of order Order
    QuadQuadrilateralWeights[Order] = quadrature weights for Gauss-Legendre quadrature of order Order
'''
QuadQuadrilateralPoints = {}
QuadQuadrilateralWeights = {}
# Obtain from line segment quadrature
for Order in QuadLinePoints.keys():
	# Extract quadrature info for reference line segment
	xqline = QuadLinePoints[Order]
	wqline = QuadLineWeights[Order]
	nqline = len(xqline)
	nquad = nqline**2
	wquad = np.zeros([nquad,1])
	xquad = np.zeros([nquad,2])
	iq = 0
	for j in range(nqline):
		xqj = xqline[j]
		wqj = wqline[j]
		for i in range(nqline):
			xqi = xqline[i]
			wqi = wqline[i]

			wquad[iq] = wqi*wqj
			xquad[iq,0] = xqi
			xquad[iq,1] = xqj

	# Store in dictionaries
	QuadQuadrilateralPoints[Order] = xquad
	QuadQuadrilateralWeights[Order] = wquad







