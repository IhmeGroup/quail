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
#       File : src/processing/readwritedatafiles.py
#
#       Contains functions for reading and writing data files.
#
# ------------------------------------------------------------------------ #
import pickle

def write_data_file(solver, iwrite):
	'''
	This function writes a data file (pickle format).

	Inputs:
	-------
	    solver: solver object
	    iwrite: integer to label data file
	'''
	
	# Remove un-pickle-able functions, objects, etc...
	solver.physics.gas = None

	# Get file name
	prefix = solver.params["Prefix"]
	if iwrite >= 0:
		fname = prefix + "_" + str(iwrite) + ".pkl"
	else:
		fname = prefix + "_final" + ".pkl"

	with open(fname, 'wb') as fo:
		# Save solver
		pickle.dump(solver, fo, pickle.HIGHEST_PROTOCOL)


def read_data_file(fname):
	'''
	This function reads a data file (pickle format).

	Inputs:
	-------
	    fname: file name (str)

	Outputs:
	--------
	    solver: solver object
	'''
	# Open and get solver
	with open(fname, 'rb') as fo:
		solver = pickle.load(fo)

	return solver
