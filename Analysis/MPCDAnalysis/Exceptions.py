class FileFormatException(Exception):
	"""
	Exception that signals that a file was not of the expected format.
	"""

	pass

class InvalidArgumentException(Exception):
	"""
	Exception that signals that an argument passed to a function was bad.
	"""

	pass

class OutOfRangeException(InvalidArgumentException):
	"""
	Exception that signals that a passed argument was out of its valid range.
	"""

	pass
