import pickle
import code

def write_data_file(solver, iwrite):
	# Unpack
	# mesh = solver.mesh
	# physics = solver.physics
	# Params = solver.Params
	# Time = solver.Time

	prefix = solver.Params["Prefix"]
	if iwrite >= 0:
		fname = prefix + "_" + str(iwrite) + ".pkl"
	else:
		fname = prefix + "_final" + ".pkl"

	with open(fname, 'wb') as fo:
		# solver.Params["RestartFile"] = fo.name
		# solver.Params["StartTime"] = Time

		pickle.dump(solver, fo, pickle.HIGHEST_PROTOCOL)
		# mesh
		# pickle.dump(mesh, fo, pickle.HIGHEST_PROTOCOL)
		# # physics
		# pickle.dump(physics, fo, pickle.HIGHEST_PROTOCOL)
		# # Params
		# pickle.dump(Params, fo, pickle.HIGHEST_PROTOCOL)
		# # Time
		# pickle.dump(Time, fo, pickle.HIGHEST_PROTOCOL)


def read_data_file(fname):

	with open(fname, 'rb') as fo:
	    solver = pickle.load(fo)

	return solver
		# mesh
	 #    mesh = pickle.load(fo)
		# # physics
	 #    physics = pickle.load(fo)
		# # Params
	 #    Params = pickle.load(fo)
		# # Time
	 #    Time = pickle.load(fo)

	# return mesh, physics, Params, Time
