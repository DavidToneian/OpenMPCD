# coding=utf-8

import math

class OnTheFlyStatistics:
	"""
	Computes the sample mean and variance of provided data incrementally.

	The algorithm used is described in the "updating formulas" (1.3) in the paper
	"Algorithms for Computing the Sample Variance: Analysis and Recommendations"
	by Tony F. Chan, Gene H. Golub, and Randall J. LeVeque,
	The American Statistician, August 1983, Vol. 37, No. 3, pp. 242-247
	DOI: 10.2307/2683386

	See also "Formulas for Robust, One-Pass Parallel Computation of Covariances
	and Arbitrary-Order Statistical Moments" by Philippe Pébay, Sandia Report
	SAND2008-6212, 2008.
	"""

	def __init__(self, mean = None, sampleVariance = None, sampleSize = None):
		"""
		The constructor.

		The parameters `mean`, `sampleVariance`, and `sampleSize` can either be
		`None`, in which case an instance without any data points is
		constructed, or all three parameters can be specified, in which case
		the instance is constructed as if data were given that result in the
		specified parameters.
		"""

		parameters = [mean, sampleVariance, sampleSize]

		if mean is None:
			for parameter in parameters:
				if parameter is not None:
					raise Exception()

			self.n = 0
			self.M = 0.0
			self.S = 0.0

			return

		for parameter in parameters:
			if parameter is None:
				raise Exception()

		self.n = sampleSize
		self.M = mean
		self.S = sampleVariance * (self.n - 1)


	def addDatum(self, datum):
		"""
		Adds a datum to the sample.
		"""

		delta = datum - self.M

		self.n = self.n + 1
		self.M = self.M + delta / self.n
		self.S = self.S + delta * (datum - self.M)


	def mergeSample(self, sample):
		"""
		Merges the given sample with this one.

		This assumes that both samples are drawn from the same population.
		"""

		if self.n == 0:
			self.n = sample.n
			self.M = sample.M
			self.S = sample.S
			return

		if self.n == 1:
			myValue = self.M

			self.n = sample.n
			self.M = sample.M
			self.S = sample.S

			self.addDatum(myValue)
			return

		if sample.n == 0:
			return

		if sample.n == 1:
			self.addDatum(sample.getSampleMean())
			return

		mergedSampleSize = 0
		for s in [self, sample]:
			mergedSampleSize += s.getSampleSize()

		mergedMean = 0
		for s in [self, sample]:
			mergedMean += s.getSampleSize() * s.getSampleMean()
		mergedMean /= mergedSampleSize

		mergedS = 0
		for s in [self, sample]:
			mergedS += (s.getSampleSize() - 1) * s.getSampleVariance()

		n = self.getSampleSize()
		m = sample.getSampleSize()
		c = (self.getSampleMean() - sample.getSampleMean())
		c /= n + m
		c **= 2
		c *= n * m * (n + m)
		mergedS += c

		self.n = mergedSampleSize
		self.M = mergedMean
		self.S = mergedS


	def mergeSamples(self, samples):
		"""
		Merges the given samples with this one.

		This assumes that all samples are drawn from the same population.
		"""

		for sample in samples:
			self.mergeSample(sample)


	def getSampleSize(self):
		"""
		Returns the number of data points added so far.
		"""

		return self.n

	def getSampleMean(self):
		"""
		Returns the mean of all the values added so far.

		If no values have been added so far, an exception is thrown.
		"""

		if self.n == 0:
			raise ValueError("Tried to get mean without having supplied data.")

		return self.M

	def getSampleVariance(self):
		"""
		Returns the unbiased sample variance of all the values added so far.

		The returned value contains Bessel's correction, i.e. the sum of
		squares of differences is divided by \f$ n - 1 \f$ rather than
		\f$ n \f$, where \f$ n \f$ is the sample size.

		If fewer than two values have been added so far, an exception is thrown.
		"""

		if self.n <= 1:
			raise ValueError("Tried to get variance without having supplied enough data.")

		return self.S / (self.n - 1)

	def getSampleStandardDeviation(self):
		"""
		Returns the unbiased sample standard deviation of all the values added
		so far.

		The returned value contains Bessel's correction, i.e. the sum of
		squares of differences is divided by \f$ n - 1 \f$ rather than
		\f$ n \f$, where \f$ n \f$ is the sample size.

		If fewer than two values have been added so far, an exception is thrown.
		"""

		return math.sqrt(self.getSampleVariance())


	def getStandardErrorOfTheMean(self):
		"""
		Returns the standard error of the mean, i.e. the unbiased sample
		standard deviation divided by the square root of the sample size.
		"""

		return \
			self.getSampleStandardDeviation() / math.sqrt(self.getSampleSize())


	def serializeToString(self):
		"""
		Returns a `str` that contains the state of this instance.

		@see unserializeFromString
		"""

		ret = ""

		ret += "1;" #format version
		ret += str(self.n) + ";"
		ret += str(self.M) + ";"
		ret += str(self.S)

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

		parts = state.split(";")
		if int(parts[0]) != 1:
			raise ValueError()

		if len(parts) != 4:
			raise ValueError()

		n = int(parts[1])
		M = float(parts[2])
		S = float(parts[3])

		if n < 0:
			raise ValueError()
		if n == 0:
			if M != 0 or S != 0:
				raise ValueError()
		if S < 0:
			raise ValueError()

		self.n = n
		self.M = M
		self.S = S


	def __eq__(self, rhs):
		"""
		The equality operator.

		Returns whether the state of this object is the same as the state of the
		`rhs` object; if `rhs` is not of this instance's type, `NotImplemented`
		is returned.

		@param[in] rhs The right-hand-side instance to compare to.
		"""

		if not isinstance(rhs, self.__class__):
			return NotImplemented

		return self.__dict__ == rhs.__dict__


	def __ne__(self, rhs):
		"""
		The inequality operator.

		Returns whether the state of this object differs from the state of the
		`rhs` object; if `rhs` is not of this instance's type, `NotImplemented`
		is returned.

		@param[in] rhs The right-hand-side instance to compare to.
		"""

		if not isinstance(rhs, self.__class__):
			return NotImplemented

		return not self == rhs


	def __hash__(self):
		"""
		Returns a hash of this object that depends on its state, and nothing
		more.
		"""

		return hash(tuple(sorted(self.__dict__.items())))


	def __repr__(self):
		"""
		Returns a `str` describing the data of this instance.
		"""

		ret = "OnTheFlyStatistics: {"
		ret += "Size: " + str(self.getSampleSize())
		if self.getSampleSize() >= 1:
			ret += ", Mean: " + str(self.getSampleMean())
			if self.getSampleSize() >= 2:
				ret += ", Standard Deviation: "
				ret += str(self.getSampleStandardDeviation())
				ret += ", Variance: " + str(self.getSampleVariance())
		ret += "}\n"
		return ret
