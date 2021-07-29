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
	print(ct) # print Warning because class is optional