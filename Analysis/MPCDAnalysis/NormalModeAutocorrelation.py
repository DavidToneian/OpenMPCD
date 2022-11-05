"""
@package MPCDAnalysis.NormalModeAutocorrelation

Analysis functionality for data on normal mode autocorrelations, as produced by
`OpenMPCD::CUDA::MPCFluid::Instrumentation::NormalModeAutocorrelation`.
"""

class NormalModeAutocorrelation:
	"""
	Analysis class for data on normal mode autocorrelation, as produced by
	`OpenMPCD::CUDA::MPCFluid::Instrumentation::NormalModeAutocorrelation`.

	Unless specified otherwise, all times are measured in units of `measurement
	time`, as defined in
	`OpenMPCD::CUDA::MPCFluid::Instrumentation::NormalModeAutocorrelation`.

	@see OpenMPCD::CUDA::MPCFluid::Instrumentation::NormalModeAutocorrelation
	"""

	def __init__(self, rundirs):
		"""
		The constructor.

		@throw TypeError
		       Throws if `rundirs` is not a `string`, or a `list` of `string`s.
		@throw ValueError
		       Throws if the given `rundir` does not exist, or does not contain
		       a readable, valid `normalModeAutocorrelations.data` file.

		@param[in] rundirs
		           The run directory, as a `string`. From this directory, the
		           file `normalModeAutocorrelations.data` will be read as input.
		           Alternatively, this may be a `list` of `string` instances,
		           each of which will be treated as described above.
		"""

		if isinstance(rundirs, str):
			rundirs = [rundirs]

		if not isinstance(rundirs, list):
			raise TypeError()
		for rundir in rundirs:
			if not isinstance(rundir, str):
				raise TypeError()


		self._config = None
		self._measurements = []
		self._normalModeCount = None
		self._statistics = {}


		from MPCDAnalysis.Configuration import Configuration
		import os.path

		for rundir in rundirs:
			filepath = rundir + "/" + "normalModeAutocorrelations.data"

			if not os.path.isfile(filepath):
				raise ValueError()

			config = Configuration(rundir)
			if self._config is None:
				self._config = config
				self._autocorrelationArgumentCount = \
					self._config[
						"instrumentation.normalModeAutocorrelation." + \
						"autocorrelationArgumentCount"]
			else:
				differences = \
					self._config.getDifferencesAsFlatDictionary(config)

				ignoredKeys = \
					[
						"mpc.sweeps",
					]
				for ignoredKey in ignoredKeys:
					if ignoredKey in differences:
						del differences[ignoredKey]

				if len(differences) != 0:
					msg = "Incompatible rundirs given."
					msg += " Differences:\n"
					msg += str(differences)
					raise ValueError(msg)

			with open(filepath, "r") as f:
				self._parse(f)


	def getNormalModeCount(self):
		"""
		Returns the number of normal modes.
		"""

		assert self._normalModeCount is not None

		return self._normalModeCount


	def getMaximumMeasurementTime(self):
		"""
		Returns, in units of `measurement time`, the maximum correlation time
		that was configured to be measured, i.e. \f$ N_A - 1 \f$.
		"""

		return self._autocorrelationArgumentCount - 1


	def getAutocorrelation(self, mode, correlationTime):
		"""
		Returns an `OnTheFlyStatisticsDDDA` object that holds information on the
		sample of measured autocorrelations for normal mode index `mode` and
		correlation time `correlationTime`.

		@throw TypeError
		       Throws if any of the arguments have invalid types.
		@throw ValueError
		       Throws if any of the arguments have invalid values.

		@param[in] mode
		           The normal mode index, as an `int` in the range
		           `[0, getNormalModeCount())`.
		@param[in] correlationTime
		           The correlation time to return results for, measured in
		           This argument is to be of type `int`, non-negative, and at
		           most `getMaximumMeasurementTime()`.
		"""

		if not isinstance(mode, int):
			raise TypeError()
		if mode < 0 or mode >= self.getNormalModeCount():
			raise ValueError()

		if not isinstance(correlationTime, int):
			raise TypeError()
		if correlationTime < 0:
			raise ValueError()
		if correlationTime > self.getMaximumMeasurementTime():
			raise ValueError()

		if mode in self._statistics:
			if correlationTime in self._statistics[mode]:
				return self._statistics[mode][correlationTime]

		if mode not in self._statistics:
			self._statistics[mode] = {}

		from MPCDAnalysis.OnTheFlyStatisticsDDDA import OnTheFlyStatisticsDDDA
		stat = OnTheFlyStatisticsDDDA()


		for measurement in self._measurements[mode]:
			if correlationTime >= len(measurement):
				continue

			stat.addDatum(measurement[correlationTime])

		self._statistics[mode][correlationTime] = stat
		return self._statistics[mode][correlationTime]


	def getMPLAxes(self, mode, showEstimatedStandardDeviation = True):
		"""
		Returns an `matplotlib.axes.Axes` object that plots the normal mode
		autocorrelation of mode index `mode` against the correlation time, in
		units of `measurement time`.

		@throw TypeError
		       Throws if any of the arguments have invalid types.
		@throw ValueError
		       Throws if any of the arguments have invalid values.

		@param[in] mode
		           The normal mode index, as an `int` in the range
		           `[0, getNormalModeCount())`.
		@param[in] showEstimatedStandardDeviation
		           Whether to show, for each data point, the estimated standard
		           deviation.
		"""

		if not isinstance(mode, int):
			raise TypeError()
		if mode < 0 or mode >= self.getNormalModeCount():
			raise ValueError()
		if not isinstance(showEstimatedStandardDeviation, bool):
			raise TypeError()


		import matplotlib.figure

		figure = matplotlib.figure.Figure()
		axes = figure.add_subplot(1, 1, 1)

		times = []
		values = []
		errorbars = []
		for T in range(0, self.getMaximumMeasurementTime() + 1):
			autocorrelation = self.getAutocorrelation(mode, T)

			times.append(T)
			values.append(autocorrelation.getSampleMean())
			if showEstimatedStandardDeviation:
				errorbars.append(
					autocorrelation.getOptimalStandardErrorOfTheMean())

		if len(errorbars) == 0:
			errorbars = None


		axes.errorbar(times, values, yerr = errorbars)

		axes.set_xlabel(r'Correlation Time $ T $')
		axes.set_ylabel(r'$ C(0, T, n=' + str(mode) + ') $')

		return axes


	def _parse(self, f):
		"""
		Parses the given file `f`.

		If a file has been parsed already, this new file is treated as if it
		was a continuation of previously parsed runs. Hence, this file's first
		measurement is treated as if it followed the last file's last
		measurement.

		@param[in] f
		           The file to parse, of type `file`.
		"""

		assert isinstance(f, file)

		starting_mt0 = 0
		if self._measurements:
			starting_mt0 = len(self._measurements[0])

		for line in f:
			parts = line.split()
			if len(parts) < 3:
				raise ValueError()

			if self._normalModeCount is None:
				self._normalModeCount = len(parts) - 2
				for _ in range(0, self._normalModeCount):
					self._measurements.append([])
			else:
				if len(parts) != self._normalModeCount + 2:
					raise ValueError()

			mt0 = int(parts[0]) + starting_mt0
			mtT = int(parts[1])

			for mode in range(0, self._normalModeCount):
				autocorrelation = float(parts[2 + mode])

				if mt0 >= len(self._measurements[mode]):
					assert mt0 == len(self._measurements[mode])
					self._measurements[mode].append([])

				if mtT != len(self._measurements[mode][mt0]):
					print(mt0)
					print(mtT)
					print(len(self._measurements[mode][mt0]))
					print("")
				assert mtT == len(self._measurements[mode][mt0])
				self._measurements[mode][mt0].append(autocorrelation)
