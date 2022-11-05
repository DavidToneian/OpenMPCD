def test_fit():
	from MPCDAnalysis.Fit import Fit
	import pytest


	def oneParam(x, p):
		return p * x

	def twoParam(x, p, c):
		return p * x + c


	fit = Fit()


	with pytest.raises(TypeError):
		fit.fit(1, [1], [1])

	with pytest.raises(TypeError):
		fit.fit(oneParam, 1, [1])

	with pytest.raises(TypeError):
		fit.fit(oneParam, [1], 1)

	with pytest.raises(TypeError):
		fit.fit(oneParam, [1], [1], 1)


	with pytest.raises(ValueError):
		fit.fit(oneParam, [1, 1], [1])

	with pytest.raises(ValueError):
		fit.fit(oneParam, [1], [1, 1])

	with pytest.raises(ValueError):
		fit.fit(oneParam, [1], [1], [1, 1])


	fit.fit(oneParam, [1, 2], [2, 4])
	fit.fit(oneParam, [1, 2], [2, 4], [0.1, 0.2])

	import numpy

	fit.fit(oneParam, numpy.array([1, 2]), [2, 4])
	fit.fit(oneParam, [1, 2], numpy.array([2, 4]))
	fit.fit(oneParam, numpy.array([1, 2]), numpy.array([2, 4]))
	fit.fit(oneParam, numpy.array([1, 2]), [2, 4], [0.1, 0.2])
	fit.fit(oneParam, [1, 2], numpy.array([2, 4]), [0.1, 0.2])
	fit.fit(oneParam, [1, 2], [2, 4], numpy.array([0.1, 0.2]))
	fit.fit(oneParam, numpy.array([1, 2]), numpy.array([2, 4]), [0.1, 0.2])
	fit.fit(oneParam, numpy.array([1, 2]), [2, 4], numpy.array([0.1, 0.2]))
	fit.fit(oneParam, [1, 2], numpy.array([2, 4]), numpy.array([0.1, 0.2]))
	fit.fit(
		oneParam,
		numpy.array([1, 2]), numpy.array([2, 4]), numpy.array([0.1, 0.2]))

	fit.fit(twoParam, [1, 2], [2, 4])



def test_fit_bounds():
	from MPCDAnalysis.Fit import Fit
	from MPCDAnalysis.Exceptions import OutOfRangeException
	import pytest


	def oneParam(x, p):
		if p < 0 or p > 7:
			raise OutOfRangeException()
		return p * x

	def twoParam(x, p, c):
		if p < 0 or p > 5:
			raise OutOfRangeException()
		if c < 3 or c > 10:
			raise OutOfRangeException()
		return p * x + c


	fit = Fit()


	with pytest.raises(TypeError):
		fit.fit(oneParam, [1], [1], lowerBounds = 1)
	with pytest.raises(TypeError):
		fit.fit(oneParam, [1], [1], upperBounds = 1)
	with pytest.raises(TypeError):
		fit.fit(oneParam, [1], [1], lowerBounds = 1, upperBounds = 1)

	with pytest.raises(TypeError):
		fit.fit(oneParam, [1], [1], lowerBounds = [1, 2])
	with pytest.raises(TypeError):
		fit.fit(oneParam, [1], [1], upperBounds = [1, 2])

	with pytest.raises(TypeError):
		fit.fit(twoParam, [1, 2], [1, 2], lowerBounds = [1])
	with pytest.raises(TypeError):
		fit.fit(twoParam, [1, 2], [1, 2], upperBounds = [1])


	with pytest.raises(OutOfRangeException):
		fit.fit(oneParam, [1, 2], [-2, -4], lowerBounds = None)
	fit.fit(oneParam, [1, 2], [-2, -4], lowerBounds = [0])


	with pytest.raises(OutOfRangeException):
		fit.fit(twoParam, [1, 2], [-3, -5], lowerBounds = None)
	with pytest.raises(OutOfRangeException):
		fit.fit(twoParam, [1, 2], [-3, -5], lowerBounds = [None, 3])
	with pytest.raises(OutOfRangeException):
		fit.fit(twoParam, [1, 2], [-3, -5], lowerBounds = [0, None])
	fit.fit(twoParam, [1, 2], [-3, -5], lowerBounds = [0, 3])

	with pytest.raises(OutOfRangeException):
		fit.fit(
			twoParam, [1, 2], [30, 50],
			lowerBounds = [0, 3], upperBounds = [None, 10])

	with pytest.raises(OutOfRangeException):
		fit.fit(
			twoParam, [1, 2], [30, 50],
			lowerBounds = [0, 3], upperBounds = [5, None])

	fit.fit(
		twoParam, [1, 2], [30, 50],
		lowerBounds = [0, 3], upperBounds = [5, 10])


def test_getFitParameterCount():
	from MPCDAnalysis.Fit import Fit
	import pytest


	def oneParam(x, p):
		return p * x

	def twoParam(x, p, c):
		return p * x + c


	fit = Fit()


	with pytest.raises(RuntimeError):
		fit.getFitParameterCount()


	fit.fit(oneParam, [1, 2], [2, 4])
	assert fit.getFitParameterCount() == 1
	assert fit.getFitParameterCount() == 1

	fit.fit(oneParam, [1, 2], [2, 4], [1.0, 0.1])
	assert fit.getFitParameterCount() == 1
	assert fit.getFitParameterCount() == 1

	fit.fit(twoParam, [1, 2], [2, 4])
	assert fit.getFitParameterCount() == 2
	assert fit.getFitParameterCount() == 2

	fit.fit(oneParam, [1, 2], [2, 4])
	assert fit.getFitParameterCount() == 1
	assert fit.getFitParameterCount() == 1



def test_getFitParameter():
	from MPCDAnalysis.Fit import Fit
	import pytest


	def oneParam(x, p):
		return p * x

	def twoParam(x, p, c):
		return p * x + c


	fit = Fit()


	with pytest.raises(RuntimeError):
		fit.getFitParameter(0)


	fit.fit(oneParam, [1, 2], [2, 4])

	with pytest.raises(TypeError):
		fit.getFitParameter(0.0)

	with pytest.raises(ValueError):
		fit.getFitParameter(-1)

	with pytest.raises(ValueError):
		fit.getFitParameter(1)

	assert fit.getFitParameter(0) == 2


	fit.fit(twoParam, [1, 2], [2, 4])
	assert fit.getFitParameter(0) == pytest.approx(2)
	assert fit.getFitParameter(1) == pytest.approx(0, abs = 3e-12)

	fit.fit(twoParam, [1, 2], [3, 5])
	assert fit.getFitParameter(0) == pytest.approx(2)
	assert fit.getFitParameter(1) == pytest.approx(1)


	import scipy.optimize

	testcases = \
	[
		[
			twoParam,
			[1, 2],
			[3.1, 5.5],
			None
		],
		[
			twoParam,
			[1, 2],
			[3.1, 5.5],
			[0.1, 0.5]
		],
	]

	for testcase in testcases:
		f, x, y, err = testcase

		expected = \
			scipy.optimize.curve_fit(
				f, x, y, sigma = err, absolute_sigma = True)

		fit.fit(f, x, y, err)

		for i in range(0, fit.getFitParameterCount()):
			assert fit.getFitParameter(i) == pytest.approx(expected[0][i])



def test_getFitParameterVariance():
	from MPCDAnalysis.Fit import Fit
	import MPCDAnalysis.Utilities

	import numpy
	import pytest


	def oneParam(x, p):
		return p * x

	def twoParam(x, p, c):
		return p * x + c


	fit = Fit()


	with pytest.raises(RuntimeError):
		fit.getFitParameterVariance(0)


	fit.fit(oneParam, [1, 2], [2, 4])

	with pytest.raises(TypeError):
		fit.getFitParameterVariance(0.0)

	with pytest.raises(ValueError):
		fit.getFitParameterVariance(-1)

	with pytest.raises(ValueError):
		fit.getFitParameterVariance(1)


	import scipy.optimize

	testcases = \
	[
		[
			twoParam,
			[1, 2],
			[2, 4],
			None
		],
		[
			twoParam,
			[1, 2],
			[3, 5],
			None
		],
		[
			twoParam,
			[1, 2],
			[3.1, 5.5],
			None
		],
		[
			twoParam,
			[1, 2],
			[3.1, 5.5],
			[0.1, 0.5]
		],

		[
			twoParam,
			[1, 2, 3],
			[2, 4, 6],
			None
		],
		[
			twoParam,
			[1, 2, 3],
			[3, 5, 7],
			None
		],
		[
			twoParam,
			[1, 2, 5],
			[3.1, 5.5, 9.9],
			None
		],
		[
			twoParam,
			[1, 2, 3.14],
			[3.1, 5.5, 7.77],
			[0.1, 0.5, 0.123]
		],
	]

	for testcase in testcases:
		f, x, y, err = testcase

		paramCount = \
			MPCDAnalysis.Utilities.getNumberOfArgumentsFromCallable(f) - 1

		estimateAvailable = paramCount < len(x) or err is not None

		import warnings
		with warnings.catch_warnings():
			if not estimateAvailable:
				warnings.simplefilter("ignore", scipy.optimize.OptimizeWarning)

			expected = \
				scipy.optimize.curve_fit(
					f, x, y, sigma = err,
					absolute_sigma = err is not None)

		fit.fit(f, x, y, err)

		for i in range(0, fit.getFitParameterCount()):
			val = expected[1][i][i]
			assert fit.getFitParameterVariance(i) == pytest.approx(val)

			if not estimateAvailable:
				assert fit.getFitParameterVariance(i) == numpy.inf


	fit.fit(oneParam, [1, 2, 3], [3, 6, 9])
	assert fit.getFitParameter(0) == 3
	assert fit.getFitParameterVariance(0) == 0

	fit.fit(oneParam, [2, 3], [6, 9])
	assert fit.getFitParameter(0) == 3
	assert fit.getFitParameterVariance(0) == 0

	fit.fit(oneParam, [3], [9])
	assert fit.getFitParameter(0) == 3
	assert fit.getFitParameterVariance(0) == numpy.inf



def test_getFitParameterStandardDeviation():
	from MPCDAnalysis.Fit import Fit
	import pytest


	def oneParam(x, p):
		return p * x

	def twoParam(x, p, c):
		return p * x + c


	fit = Fit()


	with pytest.raises(RuntimeError):
		fit.getFitParameterStandardDeviation(0)


	fit.fit(oneParam, [1, 2], [2, 4])

	with pytest.raises(TypeError):
		fit.getFitParameterStandardDeviation(0.0)

	with pytest.raises(ValueError):
		fit.getFitParameterStandardDeviation(-1)

	with pytest.raises(ValueError):
		fit.getFitParameterStandardDeviation(1)


	import numpy

	testcases = \
	[
		[
			twoParam,
			[1, 2],
			[2, 4],
			None
		],
		[
			twoParam,
			[1, 2],
			[3, 5],
			None
		],
		[
			twoParam,
			[1, 2],
			[3.1, 5.5],
			None
		],
		[
			twoParam,
			[1, 2],
			[3.1, 5.5],
			[0.1, 0.5]
		],
	]

	for testcase in testcases:
		f, x, y, err = testcase

		fit.fit(f, x, y, err)

		for i in range(0, fit.getFitParameterCount()):
			val = numpy.sqrt(fit.getFitParameterVariance(i))
			assert fit.getFitParameterStandardDeviation(i) == pytest.approx(val)


	fit.fit(oneParam, [1, 2, 3], [3, 6, 9])
	assert fit.getFitParameter(0) == 3
	assert fit.getFitParameterStandardDeviation(0) == 0

	fit.fit(oneParam, [2, 3], [6, 9])
	assert fit.getFitParameter(0) == 3
	assert fit.getFitParameterStandardDeviation(0) == 0

	fit.fit(oneParam, [3], [9])
	assert fit.getFitParameter(0) == 3
	assert fit.getFitParameterStandardDeviation(0) == numpy.inf



def test_getFitParameterCovariance():
	from MPCDAnalysis.Fit import Fit
	import MPCDAnalysis.Utilities

	import numpy
	import pytest


	def oneParam(x, p):
		return p * x

	def twoParam(x, p, c):
		return p * x + c


	fit = Fit()


	with pytest.raises(RuntimeError):
		fit.getFitParameterCovariance(0, 0)


	fit.fit(oneParam, [1, 2], [2, 4])

	with pytest.raises(TypeError):
		fit.getFitParameterCovariance(0.0, 0)

	with pytest.raises(TypeError):
		fit.getFitParameterCovariance(0, 0.0)

	with pytest.raises(TypeError):
		fit.getFitParameterCovariance(0.0, 0.0)

	with pytest.raises(ValueError):
		fit.getFitParameterCovariance(-1, 0)

	with pytest.raises(ValueError):
		fit.getFitParameterCovariance(0, -1)

	with pytest.raises(ValueError):
		fit.getFitParameterCovariance(-1, -1)

	with pytest.raises(ValueError):
		fit.getFitParameterCovariance(0, 1)

	with pytest.raises(ValueError):
		fit.getFitParameterCovariance(1, 0)

	with pytest.raises(ValueError):
		fit.getFitParameterCovariance(1, 1)


	import scipy.optimize

	testcases = \
	[
		[
			twoParam,
			[1, 2],
			[2, 4],
			None
		],
		[
			twoParam,
			[1, 2],
			[3, 5],
			None
		],
		[
			twoParam,
			[1, 2],
			[3.1, 5.5],
			None
		],
		[
			twoParam,
			[1, 2],
			[3.1, 5.5],
			[0.1, 0.5]
		],

		[
			twoParam,
			[1, 2, 3],
			[2, 4, 6],
			None
		],
		[
			twoParam,
			[1, 2, 3],
			[3, 5, 7],
			None
		],
		[
			twoParam,
			[1, 2, 5],
			[3.1, 5.5, 9.9],
			None
		],
		[
			twoParam,
			[1, 2, 3.14],
			[3.1, 5.5, 7.77],
			[0.1, 0.5, 0.123]
		],
	]

	for testcase in testcases:
		f, x, y, err = testcase

		paramCount = \
			MPCDAnalysis.Utilities.getNumberOfArgumentsFromCallable(f) - 1

		estimateAvailable = paramCount < len(x) or err is not None


		fit.fit(f, x, y, err)

		import warnings
		with warnings.catch_warnings():
			if not estimateAvailable:
				warnings.simplefilter("ignore", scipy.optimize.OptimizeWarning)

			expected = \
				scipy.optimize.curve_fit(
					f, x, y, sigma = err,
					absolute_sigma = err is not None)

		for i in range(0, fit.getFitParameterCount()):
			for j in range(0, fit.getFitParameterCount()):
				cov_ij = fit.getFitParameterCovariance(i, j)
				cov_ji = fit.getFitParameterCovariance(j, i)

				assert cov_ij == pytest.approx(cov_ji)
				assert cov_ij == pytest.approx(expected[1][i][j])

				if not estimateAvailable:
					assert cov_ij == numpy.inf
					assert cov_ji == numpy.inf



def test_multipleIndependentDataFits():
	from MPCDAnalysis.Fit import Fit
	import pytest


	def twoParam(x, p, c):
		return p * x + c


	testcases = \
	[
		[
			[1, 2],
			[2, 4],
			None
		],
		[
			[1, 2],
			[3, 5],
			None
		],
		[
			[1, 2],
			[3.1, 5.5],
			None
		],
		[
			[1, 2],
			[3.1, 5.5],
			[0.1, 0.5]
		],
	]

	results = Fit.multipleIndependentDataFits(twoParam, testcases)

	assert isinstance(results, list)

	for index, result in enumerate(results):
		assert isinstance(result, dict)
		assert result['data'] == testcases[index]

		fit = Fit()
		fit.fit(
			twoParam,
			testcases[index][0], testcases[index][1], testcases[index][2])
		assert result['fit'] == fit



def test_getBestOutOfMultipleIndependentDataFits():
	from MPCDAnalysis.Fit import Fit
	import pytest


	def twoParam(x, p, c):
		return p * x + c


	testcases = \
	[
		[
			[1, 2],
			[2, 4],
			None
		],
		[
			[1, 2],
			[3, 5],
			None
		],
		[
			[1, 2],
			[3.1, 5.5],
			None
		],
		[
			[1, 2],
			[3.1, 5.5],
			[0.1, 0.5]
		],
	]


	def comp(old, new):
		oldVariance = old["fit"].getFitParameterVariance(0)
		newVariance = new["fit"].getFitParameterVariance(0)
		return newVariance < oldVariance

	assert \
		Fit.getBestOutOfMultipleIndependentDataFits(twoParam, [], comp) is None

	fits = Fit.multipleIndependentDataFits(twoParam, testcases)

	best = fits[0]
	for i in range(1, len(fits)):
		if comp(best, fits[i]):
			best = fits[i]

	result = \
		Fit.getBestOutOfMultipleIndependentDataFits(twoParam, testcases, comp)
	assert best == result

	minVariance = 1e300
	for fit in fits:
		minVariance = min(minVariance, fit["fit"].getFitParameterVariance(0))
	assert minVariance == result["fit"].getFitParameterVariance(0)



	xData = []
	yData = []
	for x in range(-3, 3 + 1):
		xData.append(x)
		yData.append(x * x - 1)


	def oneParam(x, k):
		return x * k

	def discardFrontGenerator(xData, yData, minNumElements):
		for i in range(0, len(xData) - minNumElements + 1):
			yield [xData[i:], yData[i:], None]


	def comp(old, new):
		oldVariance = old["fit"].getFitParameterVariance(0)
		newVariance = new["fit"].getFitParameterVariance(0)
		return newVariance < oldVariance


	result = \
		Fit.getBestOutOfMultipleIndependentDataFits(
			oneParam,
			discardFrontGenerator(xData, yData, 1),
			comp)

	#expected results are taken reference data:
	assert \
		result["data"] == [xData[len(xData) - 4:], yData[len(yData) - 4:], None]
	assert result["fit"].getFitParameter(0) == 2.1428571428596355
	assert result["fit"].getFitParameterVariance(0) == 0.2312925188464976


	result = \
		Fit.getBestOutOfMultipleIndependentDataFits(
			twoParam,
			discardFrontGenerator(xData, yData, 2),
			comp)

	#expected results are taken reference data:
	assert \
		result["data"] == [xData[len(xData) - 3:], yData[len(yData) - 3:], None]
	assert result["fit"].getFitParameter(0) == 4.0000000000000009
	assert result["fit"].getFitParameterVariance(0) == 0.33333333333587728
	assert result["fit"].getFitParameter(1) == -4.3333333333333339
	assert result["fit"].getFitParameterVariance(1) == 1.5555555698350492



def test___eq__():
	from MPCDAnalysis.Fit import Fit
	import pytest

	empty1 = Fit()
	empty2 = Fit()

	with pytest.raises(TypeError):
		empty1 == 1

	assert empty1 == empty1
	assert empty2 == empty2

	assert empty1 == empty2
	assert empty2 == empty1

	f1 = Fit()
	f1.fit(lambda x, p: p * x, [1], [2])

	f2 = Fit()
	f2.fit(lambda x, p: p * x, [1], [2])

	f3 = Fit()
	f3.fit(lambda x, p: p * x, [1], [2], [3])

	f4 = Fit()
	f4.fit(lambda x, p: p * x, [1], [3])

	f5 = Fit()
	f5.fit(lambda x, p: p * x, [2], [2])


	assert f1 == f1
	assert f1 == f2

	assert not f1 == f3
	assert not f1 == f4
	assert not f1 == f5

	assert not f3 == f4
	assert not f3 == f5

	assert not f4 == f5



def test___neq__():
	from MPCDAnalysis.Fit import Fit
	import pytest

	empty1 = Fit()
	empty2 = Fit()

	with pytest.raises(TypeError):
		empty1 != 1

	assert not empty1 != empty1
	assert not empty2 != empty2

	assert not empty1 != empty2
	assert not empty2 != empty1

	f1 = Fit()
	f1.fit(lambda x, p: p * x, [1], [2])

	f2 = Fit()
	f2.fit(lambda x, p: p * x, [1], [2])

	f3 = Fit()
	f3.fit(lambda x, p: p * x, [1], [2], [3])

	f4 = Fit()
	f4.fit(lambda x, p: p * x, [1], [3])

	f5 = Fit()
	f5.fit(lambda x, p: p * x, [2], [2])


	assert not f1 != f1
	assert not f1 != f2

	assert f1 != f3
	assert f1 != f4
	assert f1 != f5

	assert f3 != f4
	assert f3 != f5

	assert f4 != f5
