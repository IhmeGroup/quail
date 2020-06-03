import numpy as np
import numerics.quadrature.segment as qseg


def get_quadrature_points_weights(order, quad_type):

	# Get quadrature points and weights for segment
	# Only Gauss-Legendre for now
	qpts_seg, qwts_seg = qseg.get_quadrature_points_weights(order, quad_type)

	# tensor product

	# weights
	qwts = np.reshape(np.outer(qwts_seg, qwts_seg), (-1,), 'F').reshape(-1,1)

	# points
	qpts = np.zeros([qwts.shape[0],2])
	qpts[:,0] = np.tile(qpts_seg, (qpts_seg.shape[0],1)).reshape(-1)
	qpts[:,1] = np.repeat(qpts_seg, qpts_seg.shape[0], axis=0).reshape(-1)

	return qpts, qwts