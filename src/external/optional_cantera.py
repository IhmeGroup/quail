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
#       File : src/external/optional_cantera.py
#
#       Contains mock class definition for cantera module
#
# ------------------------------------------------------------------------ #
try:
	import cantera as ct

except ImportError:
	class CanteraMock():
		'''
		Defines a mock class for cantera. This ensures that users
		do not need to have cantera for quail to run successfully
		'''
		def __init__(self):
			self.one_atm = 1.
			return
		def __repr__(self):
			return 'Warning: {self.__class__.__name__} is a mock class' \
				.format(self=self)
	ct = CanteraMock()