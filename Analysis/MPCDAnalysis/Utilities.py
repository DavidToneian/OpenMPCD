def openPossiblyCompressedFile(baseFilename):
	"""
	Tries to return an object with file semantics for the given path, or a path
	closely related as described below.

	When talking about `Plain` and `BZ2` instances below, what is meant is:
	- For `python2`, `Plain` corresponds to the type `file`, and `BZ2`
	  corresponds to the type `bz2.BZ2File`.
	- For `python3`, `Plain` and `BZ2` both correspond to `_io.TextIOWrapper`.

	If `baseFilename` ends in the string `.bz2`, a `BZ2` instance
	is attempted to be returned in `rt` mode for that file path.
	Otherwise, if the concatenation of `baseFilename` and the string `.bz2`
	resolves to a file, this function tries to open it in `rt` mode and return
	the resulting `BZ2` instance.
	Otherwise, if `baseFilename` resolves to a file, `open` is called on it
	with an open mode of `rt`, and that object is returned.
	Finally, if none of the above applies, an instance of `ValueError` is
	thrown.

	@throw IOError
	       Throws if a viable candidate is found, as described above, but the
	       file could not be opened.
	@throw ValueError
	       Throws if no viable candidate file can be found.

	@param[in] baseFilename
	           A path to use to open a file, as described above.
	"""

	import bz2
	import os.path
	import sys

	if sys.version_info[0] < 3:
		def openBZ2(path):
			return bz2.BZ2File(path, "r")
	else:
		def openBZ2(path):
			return bz2.open(path, "rt")

	if baseFilename[-4:] == '.bz2':
		return openBZ2(baseFilename)

	if os.path.isfile(baseFilename + '.bz2'):
		return openBZ2(baseFilename + '.bz2')

	if os.path.isfile(baseFilename):
		return open(baseFilename, "rt")

	raise ValueError("No such file: " + baseFilename)


def readValuePairsFromFile(filename, minXValue = None):
	"""
	Returns a `collections.OrderedDict`, containing key-value-pairs read from
	the given file.

	The given `filename` is supplied to `openPossiblyCompressedFile`. Then,
	each line is read. If it consists of exactly two whitespace-separated
	sub-strings, and each can be interpreted as a `float`, let the left column,
	interpreted as a `float`, be the key, and the right column, interpreted as
	a `float` be the value; otherwise, the line is ignored.
	Then, if `minXValue` is `None`, or if the key is smaller than `minXValue`
	(as determined by the `<` operator), the key-value-pair is added to the
	dictionary to be returned; otherwise, the line is ignored.

	Finally, after treating all lines in the file as described, the dictionary
	is sorted by key and returned.

	@throw IOError
	       See `openPossiblyCompressedFile`.
	@throw ValueError
	       See `openPossiblyCompressedFile`.
	@throw ValueError
	       Throws if a key appears twice in the file (not counting ignored
	       lines).
	"""

	from collections import OrderedDict

	data = OrderedDict()

	f = openPossiblyCompressedFile(filename)
	for line in f:
		columns = line.split()
		if len(columns) != 2:
			continue
		try:
			x = float(columns[0])
			y = float(columns[1])
		except ValueError:
			continue

		if x in data:
			msg = "A point has been given twice in readValuePairsFromFile: "
			msg += str(x)
			raise ValueError(msg)

		if minXValue is not None:
			if x < minXValue:
				continue

		data[x] = y

	return OrderedDict(sorted(data.items()))

def getConfigValueAndCheckConstistency(config, valueKey, knownValue):
	"""
	Given the configuration instance `config`, fetches the value associated to
	`valueKey` and compares it to `knownValue`.

	If `knownValue` is not `None`, it is compared to the given config's value.
	If it does not match, an instance of `ValueError` is raised.

	@throw TypeError
	       Throws if any argument is of the wrong type.
	@throw KeyError
	       Throws if the given `valueKey` does not exist in the given `config`.
	@throw ValueError
	       Throws if `knownValue` is not `None`, but does not equal the given
	       config's value for the given `valueKey`.

	@param[in] config
	           An instance of `MPCDAnalysis.Configuration.Configuration`.
	@param[in] valueKey
	           The configuration element to consider, as a `str` instance.
	@param[in] knownValue
	           The known value for the given `valueKey`, or `None` if unknown.

	@return Returns the given config's value for the `valueKey`.
	"""

	from .Configuration import Configuration


	if not isinstance(config, Configuration):
		raise TypeError()
	if not isinstance(valueKey, str):
		raise TypeError()


	currentValue = config[valueKey]

	if knownValue is None:
		return currentValue

	if knownValue != currentValue:
		msg = "Key " + valueKey + " yielded value "
		msg += str(currentValue)
		msg += " rather than the expeded value "
		msg += str(knownValue)
		raise ValueError(msg)

	return knownValue


def getConsistentConfigValue(rundirs, valueKey):
	"""
	For the given `valueKey`, gets the configuration value for all the given
	`rundirs` if it is consistent, and throws otherwise.

	@throw TypeError
	       Throws if any argument is of the wrong type.
	@throw ValueError
	       Throws if not all configurations agree on the value, or the value
	       does not exist in at least one configuration.

	@param[in] rundirs
	           A `list` of `str` objects, each representing a directory from
	           which to load the file `config.txt` in an instance of
	           `MPCDAnalysis.Configuration.Configuration`.
	@param[in] valueKey
	           The configuration key to consider, as a `str`.

	@return Returns the common value.
	"""

	from .Configuration import Configuration


	if not isinstance(rundirs, list):
		raise TypeError()
	for rundir in rundirs:
		if not isinstance(rundir, str):
			raise TypeError()
	if not isinstance(valueKey, str):
		raise TypeError()


	value = None

	try:
		for rundir in rundirs:
			config = Configuration(rundir + "/config.txt")
			value = getConfigValueAndCheckConstistency(config, valueKey, value)
	except KeyError:
		raise ValueError()

	return value


def getNumberOfArgumentsFromCallable(f):
	"""
	Returns the number of arguments for the given callable `f`.

	@param[in] f
	           The callable to inspect.
	"""

	import inspect
	import sys

	if sys.version_info[0] < 3:
		args = inspect.getargspec(f).args
	else:
		args = inspect.signature(f).parameters

	return len(args)
