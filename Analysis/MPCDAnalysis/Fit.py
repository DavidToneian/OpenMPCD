"""
@package MPCDAnalysis.Fit

Provides commonly used data fitting functionality.
"""

class Fit:
	"""
	Helps fitting data to arbitrary functions, and handling the results.
	"""

	def __init__(self):
		"""
		The constructor.
		"""

		self._fitResults = None


	def fit(
		self, f, xData, yData, yErrs = None,
		lowerBounds = None, upperBounds = None):
		"""
		Fits the given data to the given fit function `f`.

		After calling this, any previous fits with this instance, and the
		corresponding fit results, are discarded.

		@throw TypeError
		       Throws if any argument is of the wrong type.
		@throw ValueError
		       Throws if `xData`, `yData`, and `yErrs` (if not `None`) have
		       different lengths.
		@throw ValueError
		       Throws if `lowerBounds` or `upperBounds` has the wrong length,
		       or invalid values.

		@param[in] f
		           The function to fit to; it must be supplied as a callable
		           object, taking, as its first parameter, an instance of
		           `numpy.ndarray` containing all the independent variable
		           values to evaluate the fit function for. This callable must
		           then return an instance of `numpy.ndarray` containing, in the
		           same order, the fit function values corresponding to the
		           elements in the first argument.
		           Any further arguments this callable takes are assumed to be
		           fit parameters that will be optimized for.
		@param[in] xData
		           An iterable containing the independent variable values to fit
		           with.
		@param[in] yData
		           An iterable containing the target values, which the fit
		           should approach as much as possible. The order must
		           correspond to `xData`.
		@param[in] yErrs
		           An iterable containing, in an order corresponding to `yData`,
		           the uncertainties (one standard deviation error) on the
		           elements on `yData`.
		@param[in] lowerBounds
		           Either `None` to have no lower bounds on the fitting
		           parameters, or a `list`, the length of which corresponds to
		           the number of fitting parameters, where each value represents
		           the lower bound for the corresponding fitting parameter.
		           Values of `None` in this list correspond to no bound for that
		           parameter.
		@param[in] upperBounds
		           As in `lowerBounds`, except that upper bounds are specified.
		"""

		if not callable(f):
			raise TypeError()

		if len(xData) != len(yData):
			raise ValueError()
		if yErrs is not None:
			if len(yErrs) != len(yData):
				raise ValueError()

		for x in [lowerBounds, upperBounds]:
			from MPCDAnalysis.Utilities import getNumberOfArgumentsFromCallable
			if x is not None:
				if not isinstance(x, list):
					raise TypeError()
				if len(x) != getNumberOfArgumentsFromCallable(f) - 1:
					raise TypeError()

		import numpy
		import scipy.optimize

		if isinstance(xData, numpy.ndarray):
			xValues = xData
		else:
			try:
				xValues = numpy.array(xData)
			except:
				xValues = numpy.array([])
				for val in xData:
					xValues = numpy.append(xValues, [val])

		if isinstance(yData, numpy.ndarray):
			yValues = yData
		else:
			try:
				yValues = numpy.array(yData)
			except:
				yValues = numpy.array([])
				for val in yData:
					yValues = numpy.append(yValues, [val])

		if yErrs is None:
			yErrors = yErrs
			absoluteErrors = False
		else:
			absoluteErrors = True

			if isinstance(yErrs, numpy.ndarray):
				yErrors = yErrs
			else:
				try:
					yErrors = numpy.array(yErrors)
				except:
					yErrors = numpy.array([])
					for val in yErrs:
						yErrors = numpy.append(yErrors, [val])


		bounds = (-numpy.inf, numpy.inf)

		if lowerBounds is not None:
			_proper = []
			for b in lowerBounds:
				if b is None:
					b = -numpy.inf
				_proper.append(b)
			bounds = (_proper, bounds[1])

		if upperBounds is not None:
			_proper = []
			for b in upperBounds:
				if b is None:
					b = numpy.inf
				_proper.append(b)
			bounds = (bounds[0], _proper)


		import warnings
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", scipy.optimize.OptimizeWarning)

			self._fitResults = \
				scipy.optimize.curve_fit(
					f,
					xValues,
					yValues,
					sigma = yErrors,
					absolute_sigma = absoluteErrors,
					bounds = bounds)


	def getFitParameterCount(self):
		"""
		Returns the number of fit parameters used in the last successful call
		to `fit`.

		@throw RuntimeError
		       Throws if `fit` has not been called successfully previously.
		"""

		if self._fitResults is None:
			raise RuntimeError()

		return len(self._fitResults[0])


	def getFitParameter(self, index):
		"""
		Returns the parameter estimate for the parameter with the given `index`.

		@throw RuntimeError
		       Throws if `fit` has not been called successfully previously.
		@throw TypeError
		       Throws if any argument is of the wrong type.
		@throw ValueError
		       Throws if `index` is out of bounds.

		@param[in] index
		           The index of the fit parameter, which must be a non-negative
		           `int` that is smaller than `getFitParameterCount()`.
		"""


		if self._fitResults is None:
			raise RuntimeError()

		if not isinstance(index, int):
			raise TypeError()

		if index < 0 or index >= self.getFitParameterCount():
			raise ValueError()

		return self._fitResults[0][index]


	def getFitParameterVariance(self, index):
		"""
		Returns the variance of the parameter estimate for the parameter with
		the given `index`, if such a variance is available, or `numpy.inf` if no
		variance estimate was possible.

		@throw RuntimeError
		       Throws if `fit` has not been called successfully previously.
		@throw TypeError
		       Throws if any argument is of the wrong type.
		@throw ValueError
		       Throws if `index` is out of bounds.

		@param[in] index
		           The index of the fit parameter, which must be a non-negative
		           `int` that is smaller than `getFitParameterCount()`.
		"""


		if self._fitResults is None:
			raise RuntimeError()

		if not isinstance(index, int):
			raise TypeError()

		if index < 0 or index >= self.getFitParameterCount():
			raise ValueError()

		return self._fitResults[1][index][index]


	def getFitParameterStandardDeviation(self, index):
		"""
		Returns the standard deviation of the parameter estimate for the
		parameter with the given `index`, if such a standard deviation is
		available, or `numpy.inf` if no standard deviation estimate was
		possible.

		@throw RuntimeError
		       Throws if `fit` has not been called successfully previously.
		@throw TypeError
		       Throws if any argument is of the wrong type.
		@throw ValueError
		       Throws if `index` is out of bounds.

		@param[in] index
		           The index of the fit parameter, which must be a non-negative
		           `int` that is smaller than `getFitParameterCount()`.
		"""

		import numpy
		return numpy.sqrt(self.getFitParameterVariance(index))


	def getFitParameterCovariance(self, index1, index2):
		"""
		Returns the covariance of the parameter estimates for the parameter with
		the given indices, if such a covariance is available, or `numpy.inf` if
		no coviariance estimate was possible.

		@throw RuntimeError
		       Throws if `fit` has not been called successfully previously.
		@throw TypeError
		       Throws if any argument is of the wrong type.
		@throw ValueError
		       Throws if either `index1` or `index2` is out of bounds.

		@param[in] index1
		           The index of one of the fit parameters, which must be a
		           non-negative `int` that is smaller than
		           `getFitParameterCount()`.
		@param[in] index2
		           The index of one of the fit parameters, which must be a
		           non-negative `int` that is smaller than
		           `getFitParameterCount()`.
		"""

		if self._fitResults is None:
			raise RuntimeError()

		if not isinstance(index1, int):
			raise TypeError()
		if not isinstance(index2, int):
			raise TypeError()

		if index1 < 0 or index1 >= self.getFitParameterCount():
			raise ValueError()
		if index2 < 0 or index2 >= self.getFitParameterCount():
			raise ValueError()

		return self._fitResults[1][index1][index2]


	@staticmethod
	def multipleIndependentDataFits(f, dataSet):
		"""
		Independently fits multiple sets of data to the given fit function `f`.

		@throw TypeError
		       Throws if any argument is of the wrong type.
		@throw ValueError
		       Throws if any argument has an invalid value.

		@param[in] f
		           The function to fit to; it must satisfy the conditions for
		           the `f` argument to `fit`.
		@param[in] dataSet
		           An iterable, with elements being lists of length `3`; the
		           list elements will be passed as the `xData`, `yData`, and
		           `yErrs` parameters to `fit`, and accordingly need to satisfy
		           the conditions there.

		@return Returns a list of dictionaries, in the order that corresponds
		        to the order in `dataSet`; each dictionary contains the
		        following elements:
		        - `data`, the list returned while iterating `dataSet`
		        - `fit`, an instance of this class, on which `fit` has been
		          called with the given `f` and the `data` elements, as
		          described in the documentation for the `dataSet` parameter.
		"""

		ret = []

		for data in dataSet:
			fit = Fit()
			fit.fit(f, data[0], data[1], data[2])
			ret.append({'data': data, 'fit': fit})

		return ret


	@staticmethod
	def getBestOutOfMultipleIndependentDataFits(f, dataSet, comparator):
		"""
		Independently fits multiple sets of data to the given fit function `f`,
		and returns the best fit, as determined by `comparator`.

		This function returns the equivalent to the result of the following
		procedure:
		First, let `results = multipleIndependentDataFits(f, dataSet)`.
		Then, if `results` is empty, return `None`. Otherweise, start by calling
		the first result in the list the `best` result, and iterate through all
		other results, with iteration variable `currentResult`. For each
		iteration, call `comparator(best, currentResult)`; if it returns `True`,
		let `best = currentResult`.
		After having completed all iterations, return `best`.

		@throw TypeError
		       Throws if any argument is of the wrong type.
		@throw ValueError
		       Throws if any argument has an invalid value.

		@param[in] f
		           The function to fit to; it must satisfy the conditions for
		           the `f` argument to `fit`.
		@param[in] dataSet
		           An iterable, with elements being lists of length `3`; the
		           list elements will be passed as the `xData`, `yData`, and
		           `yErrs` parameters to `fit`, and accordingly need to satisfy
		           the conditions there.
		@param[in] comparator
		           The callable used to compare two fits; see the main body
		           of this function documentation for details.
		"""

		results = Fit.multipleIndependentDataFits(f, dataSet)

		if not results:
			return None

		best = None
		for result in results:
			if best is None:
				best = result
				continue

			if comparator(best, result):
				best = result

		return best


	def __eq__(self, rhs):
		"""
		Returns whether this instance and `rhs` have the same state.

		@throw TypeError
		       Throws if any argument is of the wrong type.

		@param[in] rhs
		           The right-hand-side instance of this class.
		"""

		if not isinstance(rhs, self.__class__):
			raise TypeError()

		if len(self.__dict__) != len(rhs.__dict__):
			return False

		import numpy
		for k, v in self.__dict__.items():
			if k not in rhs.__dict__:
				return False

			if k == "_fitResults":
				if v is None:
					if rhs.__dict__[k] is not None:
						return False
				else:
					if len(v) != len(rhs.__dict__[k]):
						return False

					for x, y in enumerate(v):
						if numpy.any(y != rhs.__dict__[k][x]):
							return False
			else:
				print(type(v))
				if v != rhs.__dict__[k]:
					return False

		return True


	def __ne__(self, rhs):
		"""
		Returns the negation of `self == rhs`.

		@throw TypeError
		       Throws if any argument is of the wrong type.

		@param[in] rhs
		           The right-hand-side instance of this class.
		"""

		return not self == rhs
