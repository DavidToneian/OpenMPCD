"""
General class for analyzing autocorrelations.
"""

class Autocorrelation:
	"""
	Analysis class for autocorrelation of data.

	Let \f$ X \left( t \right) \f$ be a function of time \f$ t \f$, which is
	known at evenly-spaced points in time; that is, one knows
	\f$ X_n = X \left( n \Delta t \right) \f$ for
	\f$ n \in \left[0, n_{\textrm{max}} \right] \f$.

	This class calculates the autocorrelation function
	\f[ C \left( N \right) =
	    \left< X \left( 0 \right) \cdot X \left( N \Delta t \right) \right> =
	    \frac{1}{n_{\textrm{max}} + 1 - N}
	    \sum_{i = 0}^{n_{\textrm{max}} - N} X_i X_{i + N}
	\f]
	for arbitrary \f$ N \in \left[0, N_{\textrm{max}} \right] \f$, where
	\f$ N_{\textrm{max}} \f$ is specified upon construction of an instance of
	this class.

	The argument \f$ N \f$ to the autocorrelation function will be called
	"correlation time" in the context of this class. It is measured in units of
	\f$ \Delta t \f$.
	"""

	def __init__(self, maxCorrelationTime):
		"""
		The constructor.

		@throw TypeError
		       Throws if any of the arguments have invalid types.
		@throw ValueError
		       Throws if any of the arguments have invalid values.

		@param[in] maxCorrelationTime
		           The maximal correlation time to measure,
		           \f$ N_{\textrm{max}} \f$, which must be a non-negative `int`.
		"""

		if not isinstance(maxCorrelationTime, int):
			raise TypeError()

		if maxCorrelationTime < 0:
			raise ValueError()


		self._maxCorrelationTime = maxCorrelationTime
		self._correlations = []
		self._dataBuffers = []

		from .OnTheFlyStatisticsDDDA import OnTheFlyStatisticsDDDA
		for _ in range(0, maxCorrelationTime + 1):
			self._correlations.append(OnTheFlyStatisticsDDDA())
			self._dataBuffers.append([])


	def getMaxCorrelationTime(self):
		"""
		Returns the maximal correlation time \f$ N_{\textrm{max}} \f$.
		"""

		return self._maxCorrelationTime


	def correlationTimeIsAvailable(self, correlationTime):
		"""
		Returns whether enough data have been supplied for there to be data on
		the correlation function with the given correlation time \f$ N \f$.

		One needs to supply at least \f$ N + 1 \f$ data points before one can
		query the correlation function for \f$ N \f$.

		@throw TypeError
		       Throws if any of the arguments have invalid types.
		@throw ValueError
		       Throws if any of the arguments have invalid values.

		@param[in] correlationTime
		           The correlation time \f$ N \f$ as an `int` in the range
		           `[0, self.getMaxCorrelationTime()]`.
		"""

		if not isinstance(correlationTime, int):
			raise TypeError()

		if correlationTime < 0:
			raise ValueError()

		if correlationTime > self.getMaxCorrelationTime():
			raise ValueError()


		return self._correlations[0].getSampleSize() > correlationTime


	def getAutocorrelation(self, correlationTime):
		"""
		Returns an `OnTheFlyStatisticsDDDA` object that holds information on the
		sample of measured autocorrelations \f$ C \left( N \right) \f$ for the
		given correlation time `correlationTime`, \f$ N \f$.

		@throw TypeError
		       Throws if any of the arguments have invalid types.
		@throw ValueError
		       Throws if any of the arguments have invalid values.
		@throw ValueError
		       Throws if `not self.correlationTimeIsAvailable(correlationTime)`.

		@param[in] correlationTime
		           The correlation time \f$ N \f$ to return results for, as an
		           `int` in the range `[0, self.getMaxCorrelationTime()]`. Also,
		           `self.correlationTimeIsAvailable(correlationTime)` must
		           return `True` for this call to be valid.
		"""

		if not isinstance(correlationTime, int):
			raise TypeError()

		if correlationTime < 0:
			raise ValueError()

		if correlationTime > self.getMaxCorrelationTime():
			raise ValueError()

		if not self.correlationTimeIsAvailable(correlationTime):
			raise ValueError()

		return self._correlations[correlationTime]


	def addDatum(self, datum, multiplicator = None):
		"""
		Supplies a new datum \f$ X_i \f$, where \f$ i \f$ is implied to be the
		number of times `addDatum` has been called previously.

		@param[in] datum
		           The datum to add, which must be compatible with the
		           `multiplicator`.
		@param[in] multiplicator
		           A callable that takes two variables, let them be called
		           \f$ X_i \f$ and \f$ X_j \f$, which are of the type that
		           `datum` is an instance of, and returns their product
		           \f$ X_i \cdot X_j \f$ as a type that is compatible with
		           `OnTheFlyStatisticsDDDA.addDatum`.
		           If `None` is given, the default multiplication operator
		           (`datum.__mul__`) is used.
		           No guarantee is given regarding the order of the operands of
		           the multiplication operator.
		"""

		def defaultMultiplicator(first, second):
			return first.__mul__(second)

		if multiplicator is None:
			multiplicator = defaultMultiplicator

		self._correlations[0].addDatum(multiplicator(datum, datum))

		for N in range(1, self.getMaxCorrelationTime() + 1):
			if len(self._dataBuffers[N]) == N:
				multiplied = multiplicator(datum, self._dataBuffers[N][0])
				self._correlations[N].addDatum(multiplied)
				self._dataBuffers[N] = self._dataBuffers[N][1:] + [datum]
			else:
				self._dataBuffers[N].append(datum)


	def getMPLAxes(self, showEstimatedStandardDeviation = True):
		"""
		Returns an `matplotlib.axes.Axes` object that plots the autocorrelation
		function against the correlation time.

		@throw TypeError
		       Throws if any of the arguments have invalid types.
		@throw ValueError
		       Throws if any of the arguments have invalid values.

		@param[in] showEstimatedStandardDeviation
		           Whether to show, for each data point, the estimated standard
		           deviation.
		"""

		if not isinstance(showEstimatedStandardDeviation, bool):
			raise TypeError()


		import matplotlib.figure

		figure = matplotlib.figure.Figure()
		axes = figure.add_subplot(1, 1, 1)

		times = []
		values = []
		errorbars = []
		for N in range(0, self.getMaxCorrelationTime() + 1):
			if not self.correlationTimeIsAvailable(N):
				break

			autocorrelation = self.getAutocorrelation(N)

			times.append(N)
			values.append(autocorrelation.getSampleMean())
			if showEstimatedStandardDeviation:
				errorbars.append(
					autocorrelation.getOptimalStandardErrorOfTheMean())

		if len(errorbars) == 0:
			errorbars = None


		axes.errorbar(times, values, yerr = errorbars)

		axes.set_xlabel(r'Correlation Time $ N $')
		axes.set_ylabel(r'$ C(N) $')

		return axes
