#! /usr/bin/python

from MPCDAnalysis.EmpiricalDistribution import EmpiricalDistribution

from collections import OrderedDict
import math
from pylibconfig import Config
import subprocess
import sys

if len(sys.argv) < 2:
	print("Usage: " + sys.argv[0] + " [--theory=springConstant,file] rundir [rundir ...]")
	exit(1)

firstRundir = 1
theoryCurves = OrderedDict()

if sys.argv[1].startswith("--theory="):
	pair = sys.argv[1][len("--theory="):]
	springConstant, filename = pair.split(",")
	theoryCurves[springConstant] = filename

	firstRundir += 1

data = OrderedDict()
for i in range(firstRundir, len(sys.argv)):
	rundir = sys.argv[i]

	config = Config()
	config.readFile(rundir + "/config.txt")
	shearRate = config.value("boundaryConditions.shearRate")[0]
	springConstant = config.value("mpc.fluid.harmonicTrimers.springConstant1")[0]
	springConstant2 = config.value("mpc.fluid.harmonicTrimers.springConstant2")[0]

	if springConstant != springConstant2:
		raise ValueError("Trimer spring constants are different in " + rundir)

	if springConstant not in data:
		data[springConstant] = OrderedDict()

	rr = EmpiricalDistribution(rundir + "/harmonicTrimers/bond1/lengthSquaredHistogram.data")

	data[springConstant][shearRate] = math.sqrt(rr.getSampleMean())


gnuplotCommand = "set terminal wxt persist\n"
gnuplotCommand += "set xlabel 'Shear Rate'\n"
gnuplotCommand += "set ylabel 'Root Mean Square Bond Length'\n"

first = True
for springConstant, _ in data.items():
	if first:
		gnuplotCommand += "plot "
		first = False
	else:
		gnuplotCommand += ", "

	gnuplotCommand += "'-' with linespoints title 'Data k=" + str(springConstant) + "'"

for springConstant, _ in theoryCurves.items():
	gnuplotCommand += ", '-' with lines title 'Theory k=" + str(springConstant) + "'"

gnuplotCommand += "\n"

gnuplotData = ""
for springConstant, shearRates in data.items():
	for shearRate, rmsLength in shearRates.items():
		gnuplotData += str(shearRate) + " " + str(rmsLength) + "\n"

	gnuplotData += "e\n"

for springConstant, filename in theoryCurves.items():
	file = open(filename, "r")
	gnuplotData += file.read()
	gnuplotData += "e\n"

gnuplotCommand += gnuplotData

gnuplot = subprocess.Popen(['gnuplot'], stdin=subprocess.PIPE)
gnuplot.stdin.write(gnuplotCommand.encode("UTF-8"))

exit(0)
