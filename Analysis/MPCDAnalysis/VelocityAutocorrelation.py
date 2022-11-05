"""
@package MPCDAnalysis.VelocityAutocorrelation

Analysis functionality for data on (center-of-mass) velocity autocorrelation,
as produced by
`OpenMPCD::CUDA::MPCFluid::Instrumentation::VelocityAutocorrelation`.
"""
from MPCDAnalysis.OnTheFlyStatisticsDDDA import OnTheFlyStatisticsDDDA

class VelocityAutocorrelation:
	"""
	Analysis class for data on mean square displacement, as produced by
	`OpenMPCD::CUDA::MPCFluid::Instrumentation::VelocityAutocorrelation`.

	Unless specified otherwise, all times are measured in units of MPC time.

	@see OpenMPCD::CUDA::MPCFluid::Instrumentation::VelocityAutocorrelation
	"""

	def __init__(self, rundirs, minimumTime):
		"""
		The constructor.

		@throw TypeError
		       Throws if `rundir` is not a `str`, or a `list` of `str`.
		@throw ValueError
		       Throws if the given `rundir` does not exist, or does not contain
		       a readable, valid data file.
		@throw ValueError
		       Throws if `rundir` is an empty list.

		@param[in] rundirs
		           The run directory, as a `string`. From this directory, the
		           file `velocityAutocorrelations.data` will be read as input.
		           If this file does not exist,
		           `velocityAutocorrelations.data.xz` will be read instead.
		           Alternatively, this can be a list of strings, each element
		           of which specifies a directory that is treated as described
		           above.
		@param[in] minimumTime
		           Discard all measurements involving simulation times smaller
		           than this value.
		"""

		if isinstance(rundirs, str):
			rundirs = [rundirs]

		if not isinstance(rundirs, list):
			raise TypeError()
		for rundir in rundirs:
			if not isinstance(rundir, str):
				raise TypeError()
		if len(rundirs) == 0:
			raise ValueError()

		if not isinstance(minimumTime, float):
			raise TypeError()

		self._minimumTime = minimumTime


		self._measurements = []
		self._statistics = {}
		self._config = None
		self._MPCTimestep = None
		self._correlationTimes = []

		from MPCDAnalysis.OnTheFlyStatisticsDDDA import OnTheFlyStatisticsDDDA

		for rundir in rundirs:
			if not isinstance(rundir, str):
				raise TypeError()

			filepath = rundir + "/" + "velocityAutocorrelations.data"
			filepathXZ = filepath + ".xz"

			import os.path
			if not os.path.isfile(filepath) and not os.path.isfile(filepathXZ):
				raise ValueError()

			from MPCDAnalysis.Configuration import Configuration
			if self._config is None:
				self._config = Configuration(rundir)
				self._measurementTime = \
					self._config[
						"instrumentation." + \
						"velocityAutocorrelation." + \
						"measurementTime"]

				self._MPCTimestep = self._config["mpc.timestep"]

			else:
				if not self._config.isEquivalent(Configuration(rundir)):
					raise ValueError(
						"Rundirs have incompatible configurations!")

			if os.path.isfile(filepath):
				with open(filepath, "r") as f:
					self._parse(f)
			elif os.path.isfile(filepathXZ):
				import lzma
				f = lzma.LZMAFile(filepathXZ, "r")
				self._parse(f)
			else:
				raise RuntimeError()


	def getAutocorrelation(self, deltaT):
		"""
		Returns an `OnTheFlyStatisticsDDDA` object that holds information on the
		sample of measured mean square displacements for time difference
		`deltaT`.

		@throw TypeError
		       Throws if any of the arguments have invalid types.
		@throw ValueError
		       Throws if any of the arguments have invalid values.

		@param[in] deltaT
		           The time difference to return results for. This must be of
		           type `float` and an element of `getCorrelationTimes()`.
		"""

		if not isinstance(deltaT, float):
			raise TypeError()
		if deltaT not in self.getCorrelationTimes():
			raise ValueError

		return self._statistics[deltaT]


	def getCorrelationTimes(self):
		"""
		Returns a sorted list of valid correlation times.
		"""

		return self._correlationTimes


	def _getDeltaT(self, t1, t2):
		"""
		Returns a rounded time difference of `t2` and `t1`.

		The returned result is rounded such that it is (approximately to within
		floating-point precision) multiple of the MPC simulation timestep used.

		@throw TypeError
		       Throws if any of the arguments have invalid types.
		@throw ValueError
		       Throws if any of the arguments have invalid values.

		@param[in] t1
		           The first time, as a non-negative `float`.
		@param[in] t1
		           The second time, as a `float` larger than or equal to `t1`.
		"""

		if not isinstance(t1, float) or not isinstance(t2, float):
			raise TypeError()
		if t1 < 0 or t2 < t1:
			raise ValueError()

		multiple = (t2 - t1) / self._MPCTimestep
		return round(multiple) * self._MPCTimestep


	def _parse(self, f):
		"""
		Parses the given file `f`.

		If a file has been parsed already, this new file is treated as if it
		was a continuation of previously parsed runs. Hence, this file's first
		measurement is treated as if it followed the last file's last
		measurement.

		@param[in] f
		           The file to parse, of type `file` or `lzma.LZMAFile`.
		"""

		for line in f:
			parts = [float(x) for x in line.split()]
			if len(parts) != 3:
				raise ValueError()

			t1, t2, value = parts

			if t2 < self._minimumTime:
				continue

			deltaT = self._getDeltaT(t1, t2)

			if deltaT not in self._statistics:
				self._statistics[deltaT] = OnTheFlyStatisticsDDDA()

			self._statistics[deltaT].addDatum(value)

		self._correlationTimes = sorted(self._statistics.keys())
