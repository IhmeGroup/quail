import numerics.helpers.helpers as helpers

def custom_user_function(solver):
	tstart = 0.
	if solver.time>=tstart:
		# Unpack
		Uc = solver.state_coeffs
		jac = solver.physics.jac
		basis_val = solver.elem_helpers.basis_val
		Uq = helpers.evaluate_state(Uc, basis_val)
		time_hist = open('time_hist.txt', 'a')
		s = str(solver.time)
		s1 = str(Uq[0,0,0])
		s2 = str(jac[0,0,0,0])
		time_hist.write(s)
		time_hist.write(' , ')
		time_hist.write(s1)
		time_hist.write(' , ')
		time_hist.write(s2)
		time_hist.write('\n')
		time_hist.close()
