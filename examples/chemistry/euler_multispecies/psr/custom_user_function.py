import numerics.helpers.helpers as helpers
import numpy as np

def custom_user_function(solver):
	tstart = 0.
	if solver.time>=tstart:
		# Unpack
		Uc = solver.state_coeffs
		# jac2 = solver.physics.jac2
		# jac = solver.physics.jac
		basis_val = solver.elem_helpers.basis_val
		Uq = helpers.evaluate_state(Uc, basis_val)

		rho = Uq[0,0,0]
		T = solver.physics.compute_variable("Temperature", Uq)
		P = solver.physics.compute_variable("Pressure", Uq)
		YH2 = solver.physics.compute_variable("MassFractionH2", Uq)
		YOH = solver.physics.compute_variable("MassFractionOH", Uq)
		YH = solver.physics.compute_variable("MassFractionH", Uq)

		time_hist = open('time_hist.txt', 'a')
		s = str(solver.time)
		rho = str(Uq[0,0,0])
		T = str(T[0,0,0])
		P = str(P[0,0,0])
		print("time: ", solver.time, " T: ", T, " P: ", P)

		yh2 = str(YH2[0,0,0])
		yoh = str(YOH[0,0,0])
		yh = str(YH[0,0,0])
		# yo = str(Uq[0,0,3])
		# yo2 = str(Uq[0,0,4])
		# yoh = str(Uq[0,0,5])
		# yh2o = str(Uq[0,0,6])
		# yho2 = str(Uq[0,0,7])
		# yh2o2 = str(Uq[0,0,8])
		# yn2 = str(Uq[0,0,9])
		#yar = str(Uq[0,0,10])

		time_hist.write(s)
		time_hist.write(' , ')
		time_hist.write(rho)
		time_hist.write(' , ')
		time_hist.write(T)
		time_hist.write(' , ')
		time_hist.write(yh2)
		time_hist.write(' , ')
		time_hist.write(yoh)
		time_hist.write(' , ')
		time_hist.write(yh)
		# time_hist.write(' , ')
		# time_hist.write(yoh)
		# time_hist.write(' , ')
		# time_hist.write(yh2o)
		# time_hist.write(' , ')
		# time_hist.write(yho2)
		# time_hist.write(' , ')
		# time_hist.write(yh2o2)
		# time_hist.write(' , ')
		# time_hist.write(yn2)
		#time_hist.write(' , ')
		#time_hist.write(yar)
		time_hist.write('\n')
		time_hist.close()
