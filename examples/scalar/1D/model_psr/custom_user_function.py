def custom_user_function(solver):

	# Unpack
	Uq = solver.state_coeffs
	time_hist = open('time_hist.txt', 'a')
	s = str(solver.time)
	s1 = str(Uq[0,0,0])
	time_hist.write(s)
	time_hist.write(' , ')
	time_hist.write(s1)
	time_hist.write('\n')
	time_hist.close()
