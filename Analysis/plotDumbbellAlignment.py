#! /usr/bin/python

from MPCDAnalysis.DumbbellAlignment import DumbbellAlignment
from MPCDAnalysis.DumbbellUtilities import getLagrangianMultiplicatorRatioFromWeissenbergNumber
from MPCDAnalysis.StaticData import StaticData

import numpy
from pylibconfig import Config
import subprocess
import sys

if len(sys.argv) < 2:
	print("Usage: " + sys.argv[0] + " rundir [rundir ...]")
	exit(1)

data = {}

for i in range(1, len(sys.argv)):
	rundir = sys.argv[i]

	config = Config()
	config.readFile(rundir + "/config.txt")
	tau_0 = config.value("mpc.fluid.dumbbell.zeroShearRelaxationTime")

	if tau_0[1]:
		tau_0 = tau_0[0]

		shearRate = config.value("boundaryConditions.shearRate")[0]
		Wi = shearRate * tau_0
	else:
		staticData = StaticData(rundir + "/static_data.txt")
		Wi = staticData['weissenbergNumber']

	if Wi in data:
		raise RuntimeError("Two runs with the same Weissenberg number given")

	align = DumbbellAlignment(rundir)

	tmp = align.getTan2Chi()
	if tmp < 0:
		tmp *= -1
	data[Wi] = tmp


gnuplotData = ""
for Wi, tanOf2Chi in sorted(data.items()):
	gnuplotData += str(Wi) + "\t" + str(tanOf2Chi) + "\n"



firstWi = sorted(data.items())[0][0]
lastWi = sorted(data.items())[-1][0]
WiSteps = 100

gnuplotTheory = ""
for Wi in numpy.linspace(firstWi, lastWi, WiSteps):
	mu = getLagrangianMultiplicatorRatioFromWeissenbergNumber(Wi)
	prediction = 2.0 * mu / Wi
	gnuplotTheory += str(Wi) + "\t" + str(prediction) + "\n"


gnuplot = subprocess.Popen(['gnuplot'], stdin=subprocess.PIPE)
gnuplotCommand = "set terminal wxt persist\n"
gnuplotCommand += "set log\n"
gnuplotCommand += "set xlabel 'Wi'\n"
gnuplotCommand += "set ylabel 'abs(tan(2 chi))'\n"

gnuplotCommand += "plot '-' with linespoints title 'Data', "
gnuplotCommand += "'-' with lines title 'Theory'\n"
gnuplotCommand += gnuplotData
gnuplotCommand += "e\n"
gnuplotCommand += gnuplotTheory
gnuplotCommand += "e"

gnuplot.stdin.write(gnuplotCommand.encode("UTF-8"))

exit(0)
