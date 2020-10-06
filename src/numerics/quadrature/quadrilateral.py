# ------------------------------------------------------------------------ #
#
#       File : src/numerics/quadrature/quadrilateral.py
#
#       Contains functions to evaluate quadrature for quadrilateral shapes
#      
# ------------------------------------------------------------------------ #
import numpy as np
import code
import numerics.quadrature.segment as qseg


def get_quadrature_points_weights(order, quad_type, forced_pts=None):
    '''
    Calls the segment quadrature function to obtain quadrature points and 
    weights and restructures them for quadrilateral shapes

    Inputs:
    ------- 
        order: solution order
        quad_type: Enum that points to the appropriate quadrature calc
        forced_pts: [OPTIONAL] number of points if forcing nodes to be 
            equal to quad_pts is turned on

    Outputs:
    --------
        qpts: quadrature point coordinates [nq, dim]
        qwts: quadrature weights [nq, 1]
    '''
    try:
        fpts = int(np.sqrt(forced_pts))
    except:
    	fpts = None

    # get quadrature points and weights for segment		
    qpts_seg, qwts_seg = qseg.get_quadrature_points_weights(order, quad_type, fpts)

    # weights
    qwts = np.reshape(np.outer(qwts_seg, qwts_seg), (-1,), 'F').reshape(-1, 1)
    # points
    qpts = np.zeros([qwts.shape[0], 2])
    qpts[:, 0] = np.tile(qpts_seg, (qpts_seg.shape[0], 1)).reshape(-1)
    qpts[:, 1] = np.repeat(qpts_seg, qpts_seg.shape[0], axis=0).reshape(-1)

    return qpts, qwts