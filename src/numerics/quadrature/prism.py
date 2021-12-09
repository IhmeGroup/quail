# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#		<https://github.com/IhmeGroup/quail>
#       
#		Copyright (C) 2020-2021
#
#       This program is distributed under the terms of the GNU
#		General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.  
#		If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#
#		File : src/numerics/quadrature/prism.py
#
#		Contains functions to evaluate quadrature for prism shapes
#
# ------------------------------------------------------------------------ #
import numpy as np
import general
import numerics.quadrature.segment as qseg
import numerics.quadrature.triangle as qtri

def get_quadrature_points_weights(order, quad_type):
	'''
	Calls the segment / triangle quadrature function to obtain quadrature 
	points and restructures them for prism shapes

	Inputs:
	-------
		order: solution order
		quad_type: Enum that points to the appropriate quadrature calc

	Outputs:
	--------
		qpts: quadrature point coordinates [nq, ndims]
		qwts: quadrature weights [nq, 1]
	'''

	# Get quadrature points and weights for triangle
	qpts_tri, qwts_tri = qtri.get_quadrature_points_weights(order, quad_type)

	# Get quadrature points and weights for segment
	qpts_seg, qwts_seg = qseg.get_quadrature_points_weights(order, 
			general.QuadratureType.GaussLegendre)

	# Weights
	qwts = np.reshape(np.outer(qwts_tri, qwts_seg), (-1,), 
			'F').reshape(-1,1)
	
	# Points
	qpts = np.zeros([qwts.shape[0], 3])
	qpts[:, :-1] = np.tile(qpts_tri, (qpts_seg.shape[0], 1))

	# Fill the third dimension with correct segment quadrature
	for iseg in range(qpts_seg.shape[0]):
		for ib in range(qpts_tri.shape[0]):
			qpts[iseg * qpts_tri.shape[0] + ib, -1] = qpts_seg[iseg]

	return qpts, qwts
