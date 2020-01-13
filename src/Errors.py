

class Error(Exception):
    '''Base class for user-defined exceptions.'''
    pass


class FileReadError(Error):
	'''Raised when a file is not read as expected'''
	pass


class NotPhysicalError(Error):
	'''Raised when a nonphysical state is obtained'''
	pass


class IncompatibleError(Error):
	'''Raised when settings are incompatible with each other'''
	pass


