import numpy
import sys

def getLagrangianMultiplicatorRatioFromWeissenbergNumber(Wi):
	if Wi == 0:
		return 1

	coefficients = [1, -1, 0, -Wi * Wi / 6.0] #see Kowalik and Winkler's paper

	roots = numpy.roots(coefficients)

	if len(roots) != 3:
		raise RuntimeError("Unexpected number of roots in " + sys._getframe().f_code.co_name)

	if not numpy.array_equal(numpy.isreal(roots), [True, False, False]):
		raise RuntimeError("Unexpected types of roots in " + sys._getframe().f_code.co_name)

	return roots[0].real
