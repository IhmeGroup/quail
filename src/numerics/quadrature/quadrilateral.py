import numpy as np
import code
import numerics.quadrature.segment as qseg


def get_quadrature_points_weights(order, quad_type, forced_pts=None):

	# Get quadrature points and weights for segment
	try:
		fpts = int(np.sqrt(forced_pts))
	except:
		fpts = None
		
	qpts_seg, qwts_seg = qseg.get_quadrature_points_weights(order, quad_type, fpts)

	# tensor product

	# weights
	qwts = np.reshape(np.outer(qwts_seg, qwts_seg), (-1,), 'F').reshape(-1,1)

	# points
	qpts = np.zeros([qwts.shape[0],2])
	qpts[:,0] = np.tile(qpts_seg, (qpts_seg.shape[0],1)).reshape(-1)
	qpts[:,1] = np.repeat(qpts_seg, qpts_seg.shape[0], axis=0).reshape(-1)

	return qpts, qwts