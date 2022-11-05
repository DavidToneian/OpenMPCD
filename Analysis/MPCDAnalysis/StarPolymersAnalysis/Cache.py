class Cache:
	"""
	Handles caching of star polymer results.
	"""

	def __init__(self, run, writePath = None):
		"""
		The constructor.

		@param[in] run
		           The run in question, which must be an instance of `Run`.
		@param[in] writePath
		           If not `None`, a path used to write data to.
		"""

		from ..Run import Run

		if not isinstance(run, Run):
			raise TypeError()

		self._run = run
		self._writePath = writePath


	def getPostProcessingPath(self, useWritePath = False):
		"""
		Returns the post-processing path for the given run.

		@param[in] useWritePath
		           Set to `True` to use return a designated path to write data.
		"""

		if useWritePath and self._writePath is not None:
			return self._writePath

		return self._run.getPath() + "/StarPolymers/post-processing"


	def storeData(self, filename, data, metadata):
		"""
		Stores the given data and metadata in the given file name, relative to
		the post-processing path.
		"""

		import os.path

		postProcessingPath = self.getPostProcessingPath(True)

		if not os.path.exists(postProcessingPath):
			import os
			os.makedirs(postProcessingPath)

		filepath = postProcessingPath + "/" + filename

		with open(filepath + ".metadata", "w") as f:
			import yaml
			yaml.safe_dump(metadata, f)

		import collections
		if isinstance(data, collections.OrderedDict):
			self._storeDataOrderedDict(data, open(filepath, "w"))
		else:
			raise Exception("Don't know how to treat `data` type")


	def hasData(self, filename, metadata):
		"""
		Returns whether there is cached data under `filename`, with metadata
		matching the given `metadata` dictionary.

		@param[in] filename
		           The name of the cached data to query.
		@param[in] metadata
		           The expected metadata.
		"""

		import os.path
		import yaml

		filepath = self.getPostProcessingPath() + "/" + filename

		if not os.path.isfile(filepath + ".metadata"):
			return False
		if not os.path.isfile(filepath):
			return False

		storedMetadata = yaml.safe_load(open(filepath + ".metadata"))

		if not storedMetadata == metadata:
			return False

		return True


	def getDataOrderedDict(
		self, filename, metadata, keyInterpreter, valueInterpreter):
		"""
		Returns the cached data under `filename`, with metadata matching the
		given `metadata`, as an instance of `collections.OrderedDict`.

		Each line of the stored file will be split at its first whitespace; the
		part of the line left of that whitespace will be fed to `keyInterpreter`
		to serve as that line's dictionary key, while the remainder of that line
		will be fed to `valueInterpreter` to generate that key's value.

		@throw RuntimeError
		       Throws if `not self.hasData(filename, metadata)`.

		@param[in] filename
		           The name of the cached data to query.
		@param[in] metadata
		           The expected metadata.
		@param[in] keyInterpreter
		           A function that will be called on each data line's key, the
		           result of which will be used as the returned dictionary's
		           key for that line.
		@param[in] valueInterpreter
		           A function that will be called on each data line's value, the
		           result of which will be used as the returned dictionary's
		           value for that line.
		"""

		if not self.hasData(filename, metadata):
			raise RuntimeError()

		filepath = self.getPostProcessingPath() + "/" + filename

		import collections
		ret = collections.OrderedDict()
		with open(filepath, "r") as f:
			for line in f:
				key, value = line.split(None, 1)
				key = keyInterpreter(key)
				value = valueInterpreter(value)
				ret[key] = value

		return ret



	def _storeDataOrderedDict(self, data, stream):
		"""
		Stores the given data in the given stream.

		@param[in] data
		           The data, in the form of a `collections.OrderedDict`.
		@param[in] stream
		           The stream to write the data to.
		"""

		import collections
		assert isinstance(data, collections.OrderedDict)

		for key, value in data.items():
			line = str(key) + "\t" + str(value) + "\n"
			stream.write(line)
