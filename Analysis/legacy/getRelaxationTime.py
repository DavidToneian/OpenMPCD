#! /usr/bin/python

from MPCDAnalysis.Data2D import Data2D

import numpy
import scipy.optimize
import sys

if len(sys.argv) < 2:
	print("Usage: " + sys.argv[0] + " datafile [datafile ...]")
	exit(1)


fitFunction = lambda x, tau, A, c: A * numpy.exp(-x / tau) + c
results = []

for i in range(1, len(sys.argv)):
	data = Data2D(sys.argv[i]).getData()

	x = []
	y = []
	for xval, yval in data.items():
		if xval > 35:
			continue

		x.append(xval)
		y.append(yval)

	fit = scipy.optimize.curve_fit(fitFunction, numpy.array(x), numpy.array(y))
	results.append(fit[0][0])

print("Resulting tau_0 from " + str(len(results)) + " samples: " + str(sum(results) / len(results)))

exit(0)
