"""
@package MPCDAnalysis.LogicalEntityMeanSquareDisplacement

Analysis functionality for data on mean square displacement, as produced by
`OpenMPCD::CUDA::MPCFluid::Instrumentation::LogicalEntityMeanSquareDisplacement`.
"""

class LogicalEntityMeanSquareDisplacement:
	"""
	Analysis class for data on mean square displacement, as produced by
	`OpenMPCD::CUDA::MPCFluid::Instrumentation::LogicalEntityMeanSquareDisplacement`.

	Unless specified otherwise, all times are measured in units of `measurement
	time`, as defined in
	`OpenMPCD::CUDA::MPCFluid::Instrumentation::LogicalEntityMeanSquareDisplacement`.

	@see OpenMPCD::CUDA::MPCFluid::Instrumentation::LogicalEntityMeanSquareDisplacement
	"""

	def __init__(self, rundirs):
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
		           file `logicalEntityMeanSquareDisplacement.data` will be read
		           as input. If this file does not exist,
		           `logicalEntityMeanSquareDisplacement.data.xz` will be read
		           instead.
		           Alternatively, this can be a list of strings, each element
		           of which specifies a directory that is treated as described
		           above.
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


		self._measurements = []
		self._statistics = {}
		self._config = None

		from MPCDAnalysis.OnTheFlyStatisticsDDDA import OnTheFlyStatisticsDDDA

		for rundir in rundirs:
			if not isinstance(rundir, str):
				raise TypeError()

			filepath = rundir + "/" + "logicalEntityMeanSquareDisplacement.data"
			filepathXZ = filepath + ".xz"

			import os.path
			if not os.path.isfile(filepath) and not os.path.isfile(filepathXZ):
				raise ValueError()

			from MPCDAnalysis.Configuration import Configuration
			if self._config is None:
				self._config = Configuration(rundir)
				self._measurementArgumentCount = \
					self._config[
						"instrumentation." + \
						"logicalEntityMeanSquareDisplacement." + \
						"measurementArgumentCount"]

				for deltaT in range(1, self.getMaximumMeasurementTime() + 1):
					self._statistics[deltaT] = OnTheFlyStatisticsDDDA()
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


	def getMaximumMeasurementTime(self):
		"""
		Returns, in units of `measurement time`, the maximum correlation time
		that was configured to be measured, i.e. \f$ N_A \f$.
		"""

		return self._measurementArgumentCount


	def getMeanSquareDisplacement(self, deltaT):
		"""
		Returns an `OnTheFlyStatisticsDDDA` object that holds information on the
		sample of measured mean square displacements for time difference
		`deltaT`.

		@throw TypeError
		       Throws if any of the arguments have invalid types.
		@throw ValueError
		       Throws if any of the arguments have invalid values.

		@param[in] deltaT
		           The time difference to return results for, measured in
		           This argument is to be of type `int`, positive, and at
		           most `getMaximumMeasurementTime()`.
		"""

		if not isinstance(deltaT, int):
			raise TypeError()
		if deltaT <= 0:
			raise ValueError()
		if deltaT > self.getMaximumMeasurementTime():
			raise ValueError()

		assert deltaT in self._statistics
		return self._statistics[deltaT]


	def fitToData(self, function = None, minTime = None, maxTime = None):
		"""
		Fits the data to the given function, and returns the optimal function
		parameters.

		@param[in] function
		           A function object suitable to be used as the first argument
		           to `scipy.optimize.curve_fit`, or `None`, in which case a
		           function of the form \f$ y\left(x\right) = p_1 x^{p_2} \f$
		           with parameters \f$ p_1 \f$ and \f$ p_2 \f$ is used.
		@param[in] minTime
		           If not `None`, this value specifies the minimum value of the
		           measurement time that will be used for the fit.
		@param[in] maxTime
		           If not `None`, this value specifies the maximum value of the
		           measurement time that will be used for the fit.
		"""

		import numpy
		import scipy.optimize

		if minTime is None:
			minTime = 1
		if maxTime is None:
			maxTime = self.getMaximumMeasurementTime() + 1

		times = []
		values = []
		for T in range(minTime, maxTime):
			msd = self.getMeanSquareDisplacement(T)

			times.append(T)
			values.append(msd.getSampleMean())

		if function is None:
			function = lambda x, p1, p2: p1 * x ** p2

		times = numpy.array(times)
		values = numpy.array(values)
		fit = scipy.optimize.curve_fit(function, times, values)

		return fit[0]


	def getMPLAxes(
			self,
			showEstimatedStandardDeviation = True,
			lines = []):
		"""
		Returns an `matplotlib.axes.Axes` object that plots the mean square
		displacement against the diffusion time, in units of `measurement time`.

		@throw TypeError
		       Throws if any of the arguments have invalid types.
		@throw ValueError
		       Throws if any of the arguments have invalid values.

		@param[in] showEstimatedStandardDeviation
		           Whether to show, for each data point, the estimated standard
		           deviation.
		@param[in] lines
		           A list of lines to plot, in addition to the data. Each
		           element of this list represents a line and is a list
		           containing, in this order:
		             - A list containing `x` and `y` coordinates for the
		               starting point of the line,
		             - Likewise for the ending point.
		"""

		if not isinstance(showEstimatedStandardDeviation, bool):
			raise TypeError()
		if not isinstance(lines, list):
			raise TypeError()


		import matplotlib.figure

		figure = matplotlib.figure.Figure()
		axes = figure.add_subplot(1, 1, 1)

		times = []
		values = []
		errorbars = []
		for T in range(1, self.getMaximumMeasurementTime() + 1):
			msd = self.getMeanSquareDisplacement(T)

			times.append(T)
			values.append(msd.getSampleMean())
			if showEstimatedStandardDeviation:
				errorbars.append(
					msd.getOptimalStandardErrorOfTheMean())

		if len(errorbars) == 0:
			errorbars = None


		axes.errorbar(times, values, yerr = errorbars)

		for line in lines:
			x = [line[0][0], line[1][0]]
			y = [line[0][1], line[1][1]]
			axes.errorbar(x, y, yerr = None)

		axes.set_xlabel(r'Diffusion Time $ T $')
		axes.set_ylabel(r'$ C(0, T) $')

		return axes


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
			parts = line.split()
			if len(parts) != 3:
				raise ValueError()

			deltaT = int(parts[1])
			msd = float(parts[2])

			self._statistics[deltaT].addDatum(msd)
