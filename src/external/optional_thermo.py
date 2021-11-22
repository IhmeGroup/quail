# ------------------------------------------------------------------------ #
#
#       File : src/external/optional_thermo.py
#
#
# ------------------------------------------------------------------------ #
import general
import ctypes
from os.path import exists

file_exists = exists(general.cantera_lib)

if file_exists:
	import physics.chemistry.euler_multispecies.tools as thermo_tools
else:
	print("WARNING: NO EXTERNAL THERMO LIBRARY DETECTED")
	import physics.chemistry.euler_multispecies.default_tools as thermo_tools