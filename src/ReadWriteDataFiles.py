import pickle


def write_data_file(solver, iwrite):
	# Unpack
	mesh = solver.mesh
	EqnSet = solver.EqnSet
	Params = solver.Params
	Time = solver.Time

	prefix = Params["Prefix"]
	if iwrite >= 0:
		fname = prefix + str(iwrite) + ".pkl"
	else:
		fname = prefix + "_final" + ".pkl"

	with open(fname, 'wb') as fo:
		# mesh
		pickle.dump(mesh, fo, pickle.HIGHEST_PROTOCOL)
		# EqnSet
		pickle.dump(EqnSet, fo, pickle.HIGHEST_PROTOCOL)
		# Params
		pickle.dump(Params, fo, pickle.HIGHEST_PROTOCOL)
		# Time
		pickle.dump(Time, fo, pickle.HIGHEST_PROTOCOL)


def read_data_file(fname):

	with open(fname, 'rb') as fo:
		# mesh
	    mesh = pickle.load(fo)
		# EqnSet
	    EqnSet = pickle.load(fo)
		# Params
	    Params = pickle.load(fo)
		# Time
	    Time = pickle.load(fo)

	return mesh, EqnSet, Params, Time

