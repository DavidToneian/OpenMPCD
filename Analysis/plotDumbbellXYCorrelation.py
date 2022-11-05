#! /usr/bin/python

from MPCDAnalysis.DumbbellUtilities import getLagrangianMultiplicatorRatioFromWeissenbergNumber
from MPCDAnalysis.EmpiricalDistribution import EmpiricalDistribution
from MPCDAnalysis.StaticData import StaticData

import numpy
import subprocess
import sys

if len(sys.argv) < 2:
	print("Usage: " + sys.argv[0] + " rundir [rundir ...]")
	exit(1)

data = {}

for i in range(1, len(sys.argv)):
	rundir = sys.argv[i]
	staticData = StaticData(rundir + "/static_data.txt")

	if staticData['weissenbergNumber'] in data:
		raise RuntimeError("Two runs with the same Weissenberg number given")

	xy = EmpiricalDistribution(rundir + "/dumbbellBondXYHistogram.gnuplot")

	data[staticData['weissenbergNumber']] = xy.getSampleMean()


l = staticData['dumbbellEquilibriumLength']
lambda_0 = 1.5 / (l * l)
gnuplotData = ""
for Wi, value in sorted(data.items()):
	gnuplotData += str(Wi) + "\t" + str(value) + "\n"




firstWi = sorted(data.items())[0][0]
lastWi = sorted(data.items())[-1][0]
WiSteps = 100

gnuplotTheory = ""
for Wi in numpy.linspace(firstWi, lastWi, WiSteps):
	mu = getLagrangianMultiplicatorRatioFromWeissenbergNumber(Wi)
	prediction = Wi / (4 * mu * mu * lambda_0)
	gnuplotTheory += str(Wi) + "\t" + str(prediction) + "\n"


gnuplot = subprocess.Popen(['gnuplot'], stdin=subprocess.PIPE)
gnuplotCommand = "set terminal wxt persist\n"
gnuplotCommand += "set log x\n"
gnuplotCommand += "set xlabel 'Wi'\n"
gnuplotCommand += "set ylabel '<Rx Ry>'\n"

gnuplotCommand += "plot '-' with linespoints notitle \n"
gnuplotCommand += gnuplotData
gnuplotCommand += "e"

gnuplot.stdin.write(gnuplotCommand.encode("UTF-8"))

exit(0)
