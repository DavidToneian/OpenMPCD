from __future__ import print_function

class OnTheFlyStatisticsCollection:
	"""
	Represents a collection of instances of `OnTheFlyStatistics`.
	"""

	def __init__(self):
		"""
		The constructor.
		"""

		self._data = {}


	def addData(self, data):
		"""
		Adds the given data to the statistics in this instance.

		@param[in] data
		           A dictionary, the values of which will be supplied to the
		           instances of `OnTheFlyStatistics` stored at the respective
		           keys. The values may either be integers or floating point
		           values, or instances of `OnTheFlyStatistics`, in which case
		           the underlying sample will be merged with the current state.
		"""

		from .OnTheFlyStatistics import OnTheFlyStatistics

		if not isinstance(data, dict):
			raise TypeError()

		for key, value in data.items():
			allowedTypes = (int, float, OnTheFlyStatistics)
			if not isinstance(value, allowedTypes):
				raise TypeError()

			if key not in self._data:
				self._data[key] = OnTheFlyStatistics()

			if isinstance(value, OnTheFlyStatistics):
				self._data[key].mergeSample(value)
			else:
				self._data[key].addDatum(value)


	def getData(self):
		"""
		Returns the current dictionary of instances of `OnTheFlyStatistics`.
		"""

		return self._data


	def getOrderedData(self):
		"""
		Returns the current instances of `OnTheFlyStatistics`, ordered by their
		key in an instance of `collections.OrderedDict`.

		The returned value can be freely altered, without changing the state of
		this instance.
		"""

		from collections import OrderedDict

		ret = OrderedDict()
		for key in sorted(self._data):
			ret[key] = self._data[key]

		return ret


	def keepOnlyEveryNthDataPoint(self, keepEveryNth):
		"""
		Keeps only every `keepEveryNth` data point, and discards the rest.

		@param[in] keepEveryNth
		           Set to any integer value greater than `1` to discard
		           `keepEveryNth - 1` data points between data points that are
		           kept.
		           Setting this to `1` amounts to not changing this instance at
		           all.
		"""

		if not isinstance(keepEveryNth, int):
			raise TypeError()
		if keepEveryNth <= 0:
			raise ValueError()

		if keepEveryNth == 1:
			return


		fullData = self._data
		self._data = {}

		skipped = 0
		for key in sorted(fullData):
			skipped += 1
			if skipped != keepEveryNth:
				continue

			skipped = 0
			self._data[key] = fullData[key]


	def getMPLAxes(
		self,
		showStandardErrorOfTheMean = True,
		showStandardDeviation = 0.3,
		singularSampleTreatment = "raise",
		plotEveryNth = 1):
		"""
		Returns an `matplotlib.axes.Axes` object that contains the current data.

		@throw ValueError
		       Throws if both `showStandardErrorOfTheMean` and
		       `showStandardDeviation` are of floating-point type.

		@param[in] showStandardErrorOfTheMean
		           Whether to show, for each data point, the standard error of
		           the mean using errorbars.
		           Set to `False` to not show the information. Set to `True` to
		           show the information as error bars. Set to a floating-point
		           value between `0.0` and `1.0` (excluding `0.0`) to use a
		           shaded area to show the information.
		@param[in] showStandardDeviation
		           Whether to show the standard deviation as a shaded region
		           around the data points.
		           Set to `False` to not show the information. Set to `True` to
		           show the information as error bars. Set to a floating-point
		           value between `0.0` and `1.0` (excluding `0.0`) to use a
		           shaded area to show the information.
		@param[in] singularSampleTreatment
		           If neither `showStandardErrorOfTheMean` nor
		           `showStandardDeviation` are `True`, this parameter has no
		           effect. Otherwise, it controls whether an exception of type
		           `RuntimeError` is to be raised if a data point consists of a
		           sample of sample size `1`, in which case no error bars or
		           standard deviations can be computed. To cause this behavior,
		           pass the string `"raise"`.
		           If the string `"discard"`, is passed, those data points are
		           silently discarded.
		           If the string `"warn"` is passed, a warning message is
		           printed to the standard output stream if there are singular
		           samples. The affected data points are assigned a standard
		           deviation and standard error of the mean of `0`.
		           If the string `"warnAndDiscard"` is passed, a warning message
		           is printed to the standard output stream if there are
		           singular samples. Those samples will be discarded.
		           Other parameter values are not allowed in any case.
		"""

		if not isinstance(showStandardErrorOfTheMean, (bool, float)):
			raise TypeError
		if isinstance(showStandardErrorOfTheMean, float):
			if showStandardErrorOfTheMean <= 0:
				raise ValueError()
			if showStandardErrorOfTheMean > 1:
				raise ValueError()

		if not isinstance(showStandardDeviation, (bool, float)):
			raise TypeError
		if isinstance(showStandardDeviation, float):
			if showStandardDeviation <= 0:
				raise ValueError()
			if showStandardDeviation > 1:
				raise ValueError()

		if isinstance(showStandardErrorOfTheMean, bool) and \
		   isinstance(showStandardDeviation, bool) and \
		   showStandardErrorOfTheMean and showStandardDeviation:
			raise ValueError()

		singularSampleTreatmentOptions = \
			["raise", "discard", "warn", "warnAndDiscard"]
		if singularSampleTreatment not in singularSampleTreatmentOptions:
			raise ValueError()


		discardSingularSamples = False
		if showStandardErrorOfTheMean or showStandardDeviation:
			if singularSampleTreatment in ["discard", "warnAndDiscard"]:
				discardSingularSamples = True


		data = self.getOrderedData()


		import matplotlib.figure

		figure = matplotlib.figure.Figure()
		axes = figure.add_subplot(1, 1, 1)

		values = []
		singularSampleKeys = []
		for key, value in data.items():
			assert value.getSampleSize() != 0

			if value.getSampleSize() == 1:
				singularSampleKeys.append(key)
				if discardSingularSamples:
					continue
			values.append(value.getSampleMean())

		if showStandardErrorOfTheMean or showStandardDeviation:
			if singularSampleTreatment == "raise":
				if len(singularSampleKeys) != 0:
					message = "Samples with sample size 1 encountered:\n"
					message += str(singularSampleKeys)
					raise RuntimeError(message)
			if singularSampleTreatment in ["warn", "warnAndDiscard"]:
				if len(singularSampleKeys) != 0:
					print("WARNING: ", end = "")
					print(len(singularSampleKeys), end = "")
					print(" samples with sample size 1 encountered:")
					print(singularSampleKeys)
			if singularSampleTreatment in ["discard", "warnAndDiscard"]:
				for key in singularSampleKeys:
					del data[key]


		standardErrorsOfTheMean = None
		if showStandardErrorOfTheMean != False:
			standardErrorsOfTheMean = []
			for value in data.values():
				if value.getSampleSize() == 1:
					standardErrorsOfTheMean.append(0)
					continue

				standardErrorsOfTheMean.append(
					value.getStandardErrorOfTheMean())

		standardDeviations = None
		if showStandardDeviation != False:
			standardDeviations = []
			for value in data.values():
				if value.getSampleSize() == 1:
					standardDeviations.append(0)
					continue
				standardDeviations.append(value.getSampleStandardDeviation())

		errorbars = None
		if isinstance(showStandardErrorOfTheMean, bool) and \
		   showStandardErrorOfTheMean == True:
			errorbars = standardErrorsOfTheMean
		if isinstance(showStandardDeviation, bool) and \
		   showStandardDeviation == True:
			errorbars = standardDeviations

		axes.errorbar(data.keys(), values, yerr = errorbars)

		if isinstance(showStandardErrorOfTheMean, float):
			import numpy
			axes.fill_between(
				data.keys(),
				numpy.array(values) - numpy.array(standardErrorsOfTheMean),
				numpy.array(values) + numpy.array(standardErrorsOfTheMean),
				alpha = showStandardErrorOfTheMean
				)

		if isinstance(showStandardDeviation, float):
			import numpy
			axes.fill_between(
				data.keys(),
				numpy.array(values) - numpy.array(standardDeviations),
				numpy.array(values) + numpy.array(standardDeviations),
				alpha = showStandardDeviation
				)

		return axes
