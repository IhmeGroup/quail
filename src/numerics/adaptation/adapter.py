import os

import ctypes
import numpy as np


class Adapter:

	def __init__(self, solver):
		self.solver = solver

	def adapt(self):
		lib_file = os.path.dirname(os.path.realpath(__file__)) + '/libmesh_adapter.so'
		lib = ctypes.cdll.LoadLibrary(lib_file)
		lib.adapt_mesh(ctypes.c_void_p(self.solver.mesh.node_coords.ctypes.data))

		breakpoint()
