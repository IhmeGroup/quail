import numerics.helpers.helpers as helpers
import pickle
import numpy as np

def custom_user_function(solver):
	pass
#	tstart = 295900.0
	# if solver.time == 0.:
	# 	solver.solution = []
	# eps = 1.0e-8
	# tfinal = 159.99999
	# if (np.abs(solver.time % 40.) < eps) or (solver.time > tfinal):
	# 	Uc = solver.state_coeffs
	# 	basis_val = solver.elem_helpers.basis_val
	# 	Uq = helpers.evaluate_state(Uc, basis_val)

	# 	solver.solution.append(Uq[0,1,0])

	# prefix = solver.params["Prefix"]
	# fname = prefix + ".pkl"

	# with open(fname, 'wb') as fo:
	# 	# Save solver
	# 	pickle.dump(solver.solution, fo, pickle.HIGHEST_PROTOCOL)



	tstart = 0.
	if solver.time>=tstart:
		# Unpack
		Uc = solver.state_coeffs
		basis_val = solver.elem_helpers.basis_val
		Uq = helpers.evaluate_state(Uc, basis_val)

		time_hist = open('time_hist.txt', 'a')
		s = str(solver.time)
		s1 = str(Uq[0,0,0])
		time_hist.write(s)
		time_hist.write(' , ')
		time_hist.write(s1)
		time_hist.write('\n')
		time_hist.close()

	# if solver.time > 79.:
	# 	Uc = solver.state_coeffs
	# 	basis_val = solver.elem_helpers.basis_val
	# 	Uq = helpers.evaluate_state(Uc, basis_val)

	# 	print(solver.time, Uq[0,1,0])
