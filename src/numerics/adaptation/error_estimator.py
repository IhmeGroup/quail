import numpy as np

import numerics.adaptation.tools as adaptation_tools
import numerics.basis.tools as basis_tools
import numerics.helpers.helpers as numerics_helpers


class ErrorEstimator:

	def __init__(self, solver):
		self.solver = solver

class SpectralDecayEstimator(ErrorEstimator):

	def __init__(self, solver):
		super().__init__(solver)
		basis = solver.basis

		# Get quadrature data for double the order (since the function being
		# integrated is squared)
		quad_order = basis.get_quadrature_order(solver.mesh, basis.order * 2)
		self.quad_pts, self.quad_wts = basis.get_quadrature_data(quad_order)

		# Get basis values for the original and truncated basis (1 lower order)
		self.basis_val = basis.get_values(self.quad_pts)
		basis.order -= 1
		self.basis_val_trunc = basis.get_values(self.quad_pts)
		basis.order += 1

	def __call__(self):
		# Unpack
		Uc = self.solver.state_coeffs
		mesh = self.solver.mesh
		basis = self.solver.basis

		# Get Jacobian determinant at quadrature points
		djac = np.empty((mesh.num_elems, self.quad_pts.shape[0], 1))
		for elem_ID in range(mesh.num_elems):
			# Inverse Jacobian at element vertices
			djac[elem_ID], _, _ = basis_tools.element_jacobian(mesh,
					elem_ID, self.quad_pts, get_djac=True, get_jac=True,
					get_ijac=True)

		# Get solution at quadrature points, for both the original and truncated
		# basis
		Uq = numerics_helpers.evaluate_state(Uc, self.basis_val,
				skip_interp=basis.skip_interp)
		Uq_trunc = numerics_helpers.evaluate_state(Uc, self.basis_val_trunc,
				skip_interp=basis.skip_interp)
		# Get just the density
		rho = Uq[:, :, 0]
		rho_trunc = Uq_trunc[:, :, 0]

		# Get square of density, and square of difference with the truncated
		# density
		rho2 = rho**2
		diff2 = (rho - rho_trunc)**2
		# Integrate over the element
		rho2_integral = np.einsum('ij, jm, ijm -> i', rho2, self.quad_wts, djac)
		diff2_integral = np.einsum('ij, jm, ijm -> i', diff2, self.quad_wts, djac)
		# The ratio is the error estimate
		epsilon_elems = diff2_integral / rho2_integral
		# Convert element values to vertex values
		epsilon = adaptation_tools.element_to_vertex(epsilon_elems,
				self.solver.elem_helpers.vol_elems, mesh.elem_to_node_IDs,
				mesh.num_nodes)
		return epsilon
