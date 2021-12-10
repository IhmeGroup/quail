import numpy as np


def element_to_vertex(f_elems, vol, elem_to_node_IDs, num_nodes):
	'''
	Compute the volume-average of elementwise data at the mesh vertices.
	'''
	f = np.zeros(num_nodes)
	vertex_vol = np.zeros(num_nodes)
	# Loop over elements
	for elem_ID in range(f_elems.shape[0]):
		node_IDs = elem_to_node_IDs[elem_ID]
		# Volume-weighted contribution of f from this element
		f[node_IDs] += f_elems[elem_ID] * vol[elem_ID]
		# Volume contribution of this element
		vertex_vol[node_IDs] += vol[elem_ID]
	# Divide by the total volume of elements touching this vertex
	return f / vertex_vol
