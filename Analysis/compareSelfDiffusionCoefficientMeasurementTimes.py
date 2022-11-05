#! /usr/bin/python

import sys

from collections import OrderedDict
from MPCDAnalysis.Data2D import Data2D
from pylibconfig import Config
import subprocess

if len(sys.argv) < 2:
	print("Usage: " + sys.argv[0] + " rundir [rundir ...]")
	exit(1)


points = OrderedDict()
for i in range(1, len(sys.argv)):
	rundir = sys.argv[i]

	config = Config()
	config.readFile(rundir + "/config.txt")
	measurementTime = config.value("instrumentation.selfDiffusionCoefficient.measurementTime")[0]
	shearRate = config.value("boundaryConditions.shearRate")[0]

	if measurementTime not in points:
		points[measurementTime] = OrderedDict()


	data = Data2D(rundir + "/selfDiffusionCoefficient.data")
	mean = data.getArithmeticMean()
	deviation = data.getRootMeanSquaredDeviationFromArithmeticMean()

	points[measurementTime][shearRate] = [mean, deviation]

	print(str(measurementTime) + " " + str(shearRate))


gnuplotData = OrderedDict()

for measurementTime, shearRates in points.items():
	gnuplotData[measurementTime] = ""
	for shearRate, values in shearRates.items():
		gnuplotData[measurementTime] += str(shearRate) + " " + str(values[0]) + " " + str(values[1]) + "\n"

gnuplot = subprocess.Popen(['gnuplot'], stdin=subprocess.PIPE)

gnuplotCommand = "set terminal wxt persist\n"
gnuplotCommand += "set xlabel 'shear rate'\n"
gnuplotCommand += "set ylabel 'self-diffusion coefficient'\n"

first = True
for measurementTime, _ in gnuplotData.items():
	if first:
		gnuplotCommand += "plot "
		first = False
	else:
		gnuplotCommand += ", "

	gnuplotCommand += "'-' using 1:2:3 with yerrorbars title '" + str(measurementTime) + "'"

gnuplotCommand += "\n"
for _, data in gnuplotData.items():
	gnuplotCommand += data
	gnuplotCommand += "e\n"

print(gnuplotCommand)

gnuplot.stdin.write(gnuplotCommand.encode("UTF-8"))

exit(0)
