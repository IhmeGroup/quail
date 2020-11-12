# ------------------------------------------------------------------------ #
#
#       File : src/errors.py
#
#       Contains user-defined errors.
#
# ------------------------------------------------------------------------ #

class Error(Exception):
    '''
    Base class for user-defined exceptions.
    '''
    pass


class FileReadError(Error):
	'''
	Raised when a file is not read as expected.
	'''
	pass


class NotPhysicalError(Error):
	'''
	Raised when a nonphysical state is obtained.
	'''
	pass


class IncompatibleError(Error):
	'''
	Raised when settings are incompatible with each other.
	'''
	pass


class DoesNotExistError(Error):
	'''
	Raised when a specific item in, for example, a list does not exist
	.'''
	pass


