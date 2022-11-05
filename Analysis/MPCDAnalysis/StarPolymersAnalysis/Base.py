import abc

class Base(object):
	__metaclass__ = abc.ABCMeta

	"""
	Base class for analysis of star polymer simulations.
	"""

	def __init__(self, run):
		"""
		The constructor.

		@param[in] run
		           The run in question, which must be an instance of `Run`.
		"""

		from ..Run import Run

		if not isinstance(run, Run):
			raise TypeError()

		self._run = run
		self._writePath = None


	def setWritePath(self, writePath):
		"""
		Sets the path to the directory that data files are written to.

		@param[in] writePath
		           If not `None`, a path used to write data to.
		"""

		self._writePath = writePath


	def getRun(self):
		"""
		Returns the run that is being analyzed.
		"""

		return self._run


	def getValueAsFunctionOfTime(self):
		"""
		Returns the asphericity as a function of time, in the form of a
		`collections.OrderedDict`.

		This function will make use of cache results, if available.
		"""

		if self._hasDataInCache():
			return self._getDataInCache()

		data = self._computeValueAsFunctionOfTime()

		self._storeDataInCache(data)

		return data


	def updateCache(self, forced = False):
		"""
		Updates the cache, if necessary or if forced.

		@param[in] forced
		           Whether to force an update, even if the cache is up to date.
		"""

		if not forced and self._hasDataInCache():
			return

		data = self._computeValueAsFunctionOfTime()

		self._storeDataInCache(data)


	def _getCache(self):
		"""
		Returns the cache instance for this run.
		"""

		from .Cache import Cache

		return Cache(self.getRun(), self._writePath)


	def _getSnapshotsPath(self):
		"""
		Returns the path to this run's snapshots.
		"""

		return self.getRun().getPath() + "/StarPolymers/snapshots.vtf"


	def _getSnapshots(self):
		"""
		Returns an instance of `VTFSnapshotFile` for this run.

		@throw RuntimeError
		       Throws if the snapshot file does not exist.
		"""

		from ..VTFSnapshotFile import VTFSnapshotFile

		return VTFSnapshotFile(self._getSnapshotsPath(), assertReadMode = True)


	def _getStarPolymers(self):
		"""
		Returns an instance of `StarPolymers` that corresponds to this run.
		"""

		from ..StarPolymers import StarPolymers

		config = self.getRun().getConfiguration()["solute.StarPolymers"]
		return StarPolymers(config)


	def _getCacheMetadata(self):
		"""
		Returns the cache metadata.
		"""

		cacheVersion = 1

		metadata = {"cacheVersion": cacheVersion}
		return metadata


	def _hasDataInCache(self):
		"""
		Returns whether there are cached data.
		"""

		metadata = self._getCacheMetadata()
		return self._getCache().hasData(self._getCacheFilename(), metadata)


	def _storeDataInCache(self, data):
		"""
		Stores the given data in the cache.
		"""

		metadata = self._getCacheMetadata()
		self._getCache().storeData(self._getCacheFilename(), data, metadata)


	def _getCacheKeyInterpreter(self):
		"""
		Returns what is to be used as the `keyInterpreter` argument to
		`Cache.getDataOrderedDict`.
		"""

		return float


	def _getCacheValueInterpreter(self):
		"""
		Returns what is to be used as the `valueInterpreter` argument to
		`Cache.getDataOrderedDict`.
		"""

		return float


	def _getDataInCache(self):
		"""
		Returns the cached data.

		@throw RuntimeError
		       Throws if `not self._hasDataInCache()`.
		"""

		if not self._hasDataInCache():
			raise RuntimeError()

		filename = self._getCacheFilename()
		metadata = self._getCacheMetadata()

		return \
			self._getCache().getDataOrderedDict(
				filename, metadata,
				self._getCacheKeyInterpreter(),
				self._getCacheValueInterpreter())
