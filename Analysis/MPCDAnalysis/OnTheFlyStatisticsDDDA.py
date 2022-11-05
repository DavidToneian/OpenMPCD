# coding=utf-8

"""
@package MPCDAnalysis.OnTheFlyStatisticsDDDA
Defines the `OnTheFlyStatisticsDDDA` class.
"""

class OnTheFlyStatisticsDDDA:
	"""
	Computes sample means and their errors for (possibly) serially correlated
	data.

	The algorithm used is called "Dynamic Distributable Decorrelation
	Algorithm" (DDDA), and is described in
	"Efficient algorithm for “on-the-fly” error analysis of local or distributed
	serially correlated data"
	by David R. Kent IV, Richard P. Muller, Amos G. Anderson,
	William A. Goddard III, and Michael T. Feldmann,
	Journal of Computational Chemistry,
	November 2007, Vol. 28, No. 14, pp. 2309-2316,
	DOI: 10.1002/jcc.20746,
	which in turn is partly based on the article
	"Error estimates on averages of correlated data"
	by H. Flyvbjerg and H. G. Petersen,
	The Journal of Chemical Physics, July 1989, Vol. 91, No. 1, pp. 461-466
	DOI: 10.1063/1.457480
	"""

	def __init__(self):
		"""
		The constructor.
		"""

		from .OnTheFlyStatistics import OnTheFlyStatistics

		self._blocks = [OnTheFlyStatistics()]
		self._waiting = [None]


	def addDatum(self, datum):
		"""
		Adds a datum to the sample.

		It is assumed that the "time" intervals between subsequently added data
		are constant; here, "time" may, for example, refer to Molecular Dynamics
		or Monte Carlo steps.
		"""

		self._addDatum(datum, 0)


	def merge(self, rhs):
		"""
		Merges the information in `rhs` with this instance's.

		Let \f$ x_1, x_2, \ldots, x_{N_1} \f$ be the data that have been
		supplied to this instance prior to calling this function, with
		\f$ N_1 \f$ being the sample size of this instance prior to calling this
		function. Similarly, let \f$ y_1, y_2, \ldots, y_{N_2} \f$ be the data
		supplied to `rhs` previously.
		Then, after this function successfully returns, this instance's state is
		approximately as if it had started empty, and then the data
		\f$ x_1, x_2, \ldots, x_{N_1}, y_1, y_2, \ldots, y_{N_2} \f$
		had been supplied. This is true to a high degree for the state of block
		`0`; other blocks may have significantly different state due to the way
		blocks are populated with data points. The implementation is equivalent
		to the one named `Decorrelation.addition` in Appendix B of
		@cite Kent2007.

		The `rhs` instance is left unchanged (unless, of course, it is the same
		object as this instance, which is allowed).

		@throw TypeError
		       Throws if `rhs` is not of type `OnTheFlyStatisticsDDDA`.

		@param[in] rhs
		           The instance to merge into this one. Must be an instance of
		           `OnTheFlyStatisticsDDDA`.
		"""

		if not isinstance(rhs, OnTheFlyStatisticsDDDA):
			raise TypeError()


		from .OnTheFlyStatistics import OnTheFlyStatistics
		while len(rhs._blocks) > len(self._blocks):
			self._blocks.append(OnTheFlyStatistics())
			self._waiting.append(None)

		for i in range(0, len(self._blocks)):
			if i >= len(rhs._blocks):
				break
			self._blocks[i].mergeSample(rhs._blocks[i])

		for i in range(0, len(self._blocks)):
			if i >= len(rhs._blocks):
				break

			if rhs._waiting[i] is None:
				continue

			if self._waiting[i] is None:
				import copy
				self._waiting[i] = copy.deepcopy(rhs._waiting[i])
			else:
				mean = (self._waiting[i] + rhs._waiting[i]) / 2.0
				self._waiting[i] = None

				if i + 1 == len(self._blocks):
					self._blocks.append(OnTheFlyStatistics())
					self._waiting.append(None)

				self._addDatum(mean, i + 1)


	def getSampleSize(self):
		"""
		Returns the number of data points added so far.
		"""

		return self._blocks[0].getSampleSize()


	def getSampleMean(self):
		"""
		Returns the mean of all the values added so far.

		Since the mean of all values added is returned, and the sample size may
		not be a power of 2, statistics with different blocking length may
		not incorporate the same amount of information. This may lead to
		difficulties when using the error estimates of statistics of different
		block lengths to estimate the error in the entire, possibly correlated
		data set, since the statistics of different blocking lengths do not
		necessarily incorporate the same measurements.

		If no values have been added so far, `Exception` is raised.
		"""

		if self.getSampleSize() == 0:
			raise Exception("Tried to get mean without having supplied data.")

		return self._blocks[0].getSampleMean()


	def getMaximumBlockSize(self):
		"""
		Returns the largest block size for which there is at least one data
		point.

		If no values have been added so far, `Exception` is raised.
		"""

		if self.getSampleSize() == 0:
			raise Exception()

		return 2 ** self.getMaximumBlockID()


	def getMaximumBlockID(self):
		"""
		Returns the ID of the largest block size created so far.
		"""

		return len(self._blocks) - 1


	def hasBlockVariance(self, blockID):
		"""
		Returns whether the block with the given `blockID` has enough data to
		compute a sample variance.

		@param[in] blockID
		           The block ID, which must be an integer in the range
		           [0, `getMaximumBlockID()`].
		"""

		if not isinstance(blockID, int):
			raise TypeError()

		if blockID < 0 or blockID > self.getMaximumBlockID():
			raise ValueError()

		return self._blocks[blockID].getSampleSize() >= 2


	def getBlockVariance(self, blockID):
		"""
		Returns the sample variance in the block with the given `blockID`.

		@throw RuntimeError
		       Throws if `not self.hasBlockVariance(blockID)`.

		@param[in] blockID
		           The block ID, which must be an integer in the range
		           [0, `getMaximumBlockID()`].
		"""

		if not isinstance(blockID, int):
			raise TypeError()

		if blockID < 0 or blockID > self.getMaximumBlockID():
			raise ValueError()

		if not self.hasBlockVariance(blockID):
			raise RuntimeError()

		return self._blocks[blockID].getSampleVariance()


	def getBlockStandardDeviation(self, blockID):
		"""
		Returns the sample standard deviation in the block with the given
		`blockID`.

		@throw TypeError
		       Throws if `blockID` is not of type `int`.
		@throw ValueError
		       Throws if `blockID` is out of range.
		@throw RuntimeError
		       Throws if `not self.hasBlockVariance(blockID)`.

		@param[in] blockID
		           The block ID, which must be an integer in the range
		           [0, `getMaximumBlockID()`].
		"""

		import math
		return math.sqrt(self.getBlockVariance(blockID))


	def getSampleStandardDeviation(self):
		"""
		Returns the raw sample standard deviation, i.e. the sample standard
		deviation in block `0`.

		@throw RuntimeError
		       Throws if `not self.hasBlockVariance(0)`.
		"""

		return self.getBlockStandardDeviation(0)


	def getBlockStandardErrorOfTheMean(self, blockID):
		"""
		Returns an estimate for the standard deviation of the standard error of
		the mean for a given `blockID`.

		@throw RuntimeError
		       Throws if `not self.hasBlockVariance(blockID)`.

		@param[in] blockID
		           The block ID, which must be an integer in the range
		           [0, `getMaximumBlockID()`].
		"""

		if not isinstance(blockID, int):
			raise TypeError()

		if blockID < 0 or blockID > self.getMaximumBlockID():
			raise ValueError()

		if not self.hasBlockVariance(blockID):
			raise RuntimeError()

		return self._blocks[blockID].getStandardErrorOfTheMean()



	def getEstimatedStandardDeviationOfBlockStandardErrorOfTheMean(
		self, blockID):
		"""
		Returns an estimate for the standard deviation of the standard error of
		the mean for a given `blockID`.

		The returned estimate corresponds to Eq. (28) in
		"Error estimates on averages of correlated data"
		by H. Flyvbjerg and H. G. Petersen,
		The Journal of Chemical Physics, July 1989, Vol. 91, No. 1, pp. 461-466
		DOI: 10.1063/1.457480

		@throw RuntimeError
		       Throws if `not self.hasBlockVariance(blockID)`.

		@param[in] blockID
		           The block ID, which must be an integer in the range
		           [0, `getMaximumBlockID()`].
		"""

		if not isinstance(blockID, int):
			raise TypeError()

		if blockID < 0 or blockID > self.getMaximumBlockID():
			raise ValueError()

		if not self.hasBlockVariance(blockID):
			raise RuntimeError()

		se = self.getBlockStandardErrorOfTheMean(blockID)
		reducedSampleSize = self._blocks[blockID].getSampleSize()

		import math
		return se / math.sqrt(2 * reducedSampleSize)


	def getOptimalBlockIDForStandardErrorOfTheMean(self):
		"""
		Returns the block ID corresponding to the optimal block size, in the
		sense that the corresponding block provides the most accurate estimate
		for the standard error of the mean.

		If there is no variance in the data, `0` is returned.

		The algorithm used is described in Section IV of the article
		"Strategies for improving the efficiency of quantum Monte Carlo
		calculations"
		by R. M. Lee, G. J. Conduit, N. Nemec, P. López Ríos, and
		N. D. Drummond,
		Physical Review E, June 2011, Vol. 83, No. 6, pp. 066706
		DOI: 10.1103/PhysRevE.83.066706

		@throw RuntimeError
		       Throws if fewer than two data points have been supplied.
		"""

		if self.getSampleSize() < 2:
			raise RuntimeError()

		rawStandardError = \
			self._blocks[0].getStandardErrorOfTheMean()

		if rawStandardError == 0:
			return 0

		optimalBlockID = self.getMaximumBlockID()

		for blockID in range(self.getMaximumBlockID(), -1, -1):
			if not self.hasBlockVariance(blockID):
				assert blockID == self.getMaximumBlockID()

				optimalBlockID -= 1
				continue

			blockSize = 2 ** blockID
			blockedStandardError = \
				self._blocks[blockID].getStandardErrorOfTheMean()
			quotient = blockedStandardError / rawStandardError
			threshold = 2 * self.getSampleSize() * quotient ** 4
			if blockSize ** 3 > threshold:
				optimalBlockID = blockID

		return optimalBlockID


	def optimalStandardErrorOfTheMeanEstimateIsReliable(self):
		"""
		Returns whether the sample is large enough for the estimate of the
		standard error of the mean, as provided by the block indicated by
		`getOptimalBlockIDForStandardErrorOfTheMean`, to be reliable.

		The algorithm used is described in Section IV of the article
		"Strategies for improving the efficiency of quantum Monte Carlo
		calculations"
		by R. M. Lee, G. J. Conduit, N. Nemec, P. López Ríos, and
		N. D. Drummond,
		Physical Review E, June 2011, Vol. 83, No. 6, pp. 066706
		DOI: 10.1103/PhysRevE.83.066706

		@throw RuntimeError
		       Throws if fewer than two data points have been supplied.
		"""

		blockID = self.getOptimalBlockIDForStandardErrorOfTheMean()
		blockSize = 2 ** blockID

		return 50 * blockSize < self.getSampleSize()


	def getOptimalStandardErrorOfTheMean(self):
		"""
		Returns the best estimation of the true standard error of the mean of
		the data, after decorrelation.

		@see optimalStandardErrorOfTheMeanEstimateIsReliable

		The algorithm used is described in Section IV of the article
		"Strategies for improving the efficiency of quantum Monte Carlo
		calculations"
		by R. M. Lee, G. J. Conduit, N. Nemec, P. López Ríos, and
		N. D. Drummond,
		Physical Review E, June 2011, Vol. 83, No. 6, pp. 066706
		DOI: 10.1103/PhysRevE.83.066706

		@throw RuntimeError
		       Throws if fewer than two data points have been supplied.
		"""

		blockID = self.getOptimalBlockIDForStandardErrorOfTheMean()

		return self.getBlockStandardErrorOfTheMean(blockID)


	def serializeToString(self):
		"""
		Returns a `str` that contains the state of this instance.

		@see unserializeFromString
		"""

		ret = ""

		ret += "1|" #format version
		ret += str(len(self._blocks))
		for block in self._blocks:
			ret += "|" + block.serializeToString()
		for waiting in self._waiting:
			ret += "|"
			if waiting is not None:
				ret += str(waiting)

		return ret


	def unserializeFromString(self, state):
		"""
		Discards the current state, and loads the state specified in the given
		string instead.

		@throw TypeError
		       Throws if `state` is of the wrong type.
		@throw ValueError
		       Throws if `state` does not encode a valid state.

		@param[in] state
		           The state to load. Must be a `str` created by
		           `serializeToString`.
		"""

		if not isinstance(state, str):
			raise TypeError()

		parts = state.split("|")
		if int(parts[0]) != 1:
			raise ValueError()

		blockCount = int(parts[1])
		if blockCount < 0:
			raise ValueError()

		if len(parts) != 2 + 2 * blockCount:
			raise ValueError()

		from .OnTheFlyStatistics import OnTheFlyStatistics
		blocks = []
		for i in range(0, blockCount):
			block = OnTheFlyStatistics()
			block.unserializeFromString(parts[2 + i])
			blocks.append(block)

		waiting = []
		for i in range(0, blockCount):
			part = parts[2 + blockCount + i]
			if len(part) == 0:
				waiting.append(None)
			else:
				waiting.append(float(part))


		if blockCount == 0:
			blocks = [OnTheFlyStatistics()]
			waiting = [None]

		self._blocks = blocks
		self._waiting = waiting


	def getMPLAxes(self, showEstimatedStandardDeviation = True):
		"""
		Returns an `matplotlib.axes.Axes` object that plots the estimated
		standard error of the mean, and optionally its estimated standard
		deviation, as a function of the base-2 logarithm of the block size.

		@throw TypeError
		       Throws if any of the arguments have invalid types.

		@param[in] showEstimatedStandardDeviation
		           Whether to show, for each data point, the estimated standard
		           deviation of the estimated standard error of the mean.
		           Must be a `bool` value.
		"""

		if not isinstance(showEstimatedStandardDeviation, bool):
			raise TypeError


		import matplotlib.figure

		figure = matplotlib.figure.Figure()
		axes = figure.add_subplot(1, 1, 1)

		blockSizes = []
		values = []
		for blockID in range(0, self.getMaximumBlockID() + 1):
			if not self.hasBlockVariance(blockID):
				continue

			blockSizes.append(blockID)
			values.append(self.getBlockStandardErrorOfTheMean(blockID))

		errorbars = None
		if showEstimatedStandardDeviation:
			getSD = \
				self.getEstimatedStandardDeviationOfBlockStandardErrorOfTheMean
			errorbars = []
			for blockID in range(0, self.getMaximumBlockID() + 1):
				if not self.hasBlockVariance(blockID):
					continue
				errorbars.append(getSD(blockID))

		axes.errorbar(blockSizes, values, yerr = errorbars)

		axes.set_xlabel(r'$ \log_2 \left( BlockSize \right) $')
		axes.set_ylabel(r'Estimated Standard Error of the Mean')

		return axes


	def __eq__(self, rhs):
		"""
		Returns whether this instance and `rhs` are equivalent, i.e. whether the
		two instances behave as if they had been supplied the same data.

		@throw TypeError
		       Throws if `rhs` is not of type `OnTheFlyStatisticsDDDA`.

		@param[in] rhs
		           The `OnTheFlyStatisticsDDDA` instance to compare to.
		"""

		if not isinstance(rhs, self.__class__):
			raise TypeError()

		return self.__dict__ == rhs.__dict__


	def __ne__(self, rhs):
		"""
		Returns the negation of `self == rhs`.

		@throw TypeError
		       Throws if `rhs` is not of type `OnTheFlyStatisticsDDDA`.

		@param[in] rhs
		           The `OnTheFlyStatisticsDDDA` instance to compare to.
		"""

		return not self == rhs


	def __repr__(self):
		"""
		Returns a string (type `str`) representation of this instance.
		"""

		import math

		ret = "OnTheFlyStatisticsDDDA: {"
		ret += "Size: " + str(self.getSampleSize())
		if self.getSampleSize() >= 1:
			ret += ", Mean: " + str(self.getSampleMean())
			if self.getSampleSize() >= 2:
				ret += ", Standard Deviation [blockID=0]: "
				ret += str(math.sqrt(self.getBlockVariance(0)))
				ret += ", Variance [blockID=0]: "
				ret += str(self.getBlockVariance(0))
		ret += "}\n"
		return ret


	def _addDatum(self, datum, blockID):
		self._blocks[blockID].addDatum(datum)

		if self._waiting[blockID] is None:
			self._waiting[blockID] = datum
			return

		mean = (self._waiting[blockID] + datum) / 2.0
		self._waiting[blockID] = None
		if blockID + 1 == len(self._blocks):
			from .OnTheFlyStatistics import OnTheFlyStatistics
			self._blocks.append(OnTheFlyStatistics())
			self._waiting.append(None)

		self._addDatum(mean, blockID + 1)
