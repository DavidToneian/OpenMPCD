#! /usr/bin/python

from MPCDAnalysis.Data2D import Data2D
from MPCDAnalysis.DumbbellUtilities import getLagrangianMultiplicatorRatioFromWeissenbergNumber
from MPCDAnalysis.EmpiricalDistribution import EmpiricalDistribution
from MPCDAnalysis.StaticData import StaticData

from collections import OrderedDict
import errno
import numpy
import os.path
from pylibconfig import Config
import re
import subprocess
import sys

def usage():
	print("Usage: " + sys.argv[0] + " [--skip=firstTimestampToConsider] rundir [rundir ...]")
	exit(1)


def readRegularData(rundir, data):
	if not os.path.isfile(rundir + "/config.txt"):
		return False

	class OldStyleConfig:
		pass

	try:
		xx = Data2D(rundir + "/dumbbellAverageBondXXVSTime.data", startTime)
		yy = Data2D(rundir + "/dumbbellAverageBondYYVSTime.data", startTime)

		xxMean = xx.getArithmeticMean()
		yyMean = yy.getArithmeticMean()

		config = Config()
		config.readFile(rundir + "/config.txt")


		initialPositions = config.value("initialization.dumbbell.relativePosition")[0]

		l = config.value("mpc.fluid.dumbbell.rootMeanSquareLength")
		if not l[1]:
			raise OldStyleConfig
		l = l[0]

		tau_0 = config.value("mpc.fluid.dumbbell.zeroShearRelaxationTime")[0]
		shearRate = config.value("boundaryConditions.shearRate")[0]
		Wi = shearRate * tau_0

		analyticalStreaming = config.value("mpc.fluid.dumbbell.analyticalStreaming")
		mdStepCount = 0
		if analyticalStreaming[1]:
			analyticalStreaming = analyticalStreaming[0]
			if not analyticalStreaming:
				mdStepCount = config.value("mpc.fluid.dumbbell.mdStepCount")[0]
		else:
			analyticalStreaming = True

	except (IOError, OSError, OldStyleConfig):
		staticData = StaticData(rundir + "/static_data.txt")

		Wi = staticData['weissenbergNumber']
		l = staticData['dumbbellEquilibriumLength']

		initialPositions = 'relaxed'

		xx = EmpiricalDistribution(rundir + "/dumbbellBondXXHistogram.gnuplot")
		yy = EmpiricalDistribution(rundir + "/dumbbellBondYYHistogram.gnuplot")

		xxMean = xx.getSampleMean()
		yyMean = yy.getSampleMean()


	try:
		gitRevision = open(rundir + "/git-revision", 'r').readline()[:-1]
		if gitRevision == 'c80019d3fd5b78b760eba9b021ec3d200f0440f7':
			#correct the bug fixed in commit c86c8a50ef9a05e78b09b02454b1a1ebb4f74638
			xxMean *= 2
			yyMean *= 2
	except (IOError, OSError) as e:
		if e.errno != errno.ENOENT:
			raise


	yscale = 3.0 / (l * l)
	xxMean *= yscale
	yyMean *= yscale

	if not Wi in data:
		data[Wi] = []

	data[Wi].append(
		{
			'xx': xxMean,
			'yy': yyMean,
			'l': l,
			'initialPositions': initialPositions,
			'analyticalStreaming': analyticalStreaming,
			'mdStepCount': mdStepCount
		})

	return True



def readQuickAndDirtyGPUData(rundir, data):
	if not os.path.isfile(rundir + "/run.sh"):
		return False

	runFile = open(rundir + "/run.sh", 'r')
	for line in runFile:
		if "mpc_cuda_hybrid" in line:
			regex = re.compile('.*mpc_cuda_hybrid [0-9]+ [01] 0 0 ([0-9]+) 0 [0-9]+')
			match = regex.match(line)
			if match is not None:
				shearRate = float(match.group(1)) / 1000.0
				break

	tau_0 = 26.8164398539
	Wi = shearRate * tau_0
	l = 3

	if not Wi in data:
		data[Wi] = []

	stdoutFile = open(rundir + "/stdout.log", 'r')
	xxValues = []
	yyValues = []
	for line in stdoutFile:
		if "xx: " in line:
			xxValues.append(float(line[4:]))
		if "yy: " in line:
			yyValues.append(float(line[4:]))

		if len(xxValues) == 5000:
			break

	xxMean = sum(xxValues) / len(xxValues)
	yyMean = sum(yyValues) / len(yyValues)

	yscale = 3.0 / (l * l)
	xxMean *= yscale
	yyMean *= yscale

	data[Wi].append(
		{
			'xx': xxMean,
			'yy': yyMean,
			'l': l,
			'initialPositions': 'random'
		})

	return True


if len(sys.argv) < 2:
	usage()

data = {}

firstFileIndex = 1
startTime = 0

if sys.argv[1].startswith("--skip="):
	firstFileIndex = 2
	startTime = float(sys.argv[1][len("--skip="):])
	if len(sys.argv) < 3:
		usage()


for i in range(firstFileIndex, len(sys.argv)):
	rundir = sys.argv[i]

	if not readRegularData(rundir, data):
		if not readQuickAndDirtyGPUData(rundir, data):
			raise Exception("Could not read data in rundir " + rundir)


gnuplotData = OrderedDict()
for Wi, curvePoints in sorted(data.items()):
	for curvePoint in curvePoints:
		curveName = ", l=" + str(curvePoint['l'])
		curveName += ", " + curvePoint['initialPositions']
		if curvePoint['analyticalStreaming']:
			curveName += ', AS'
		else:
			curveName += ', MD ' + str(curvePoint['mdStepCount']) + ' steps'

		curveNameXX = "<x^2>" + curveName
		curveNameYY = "<y^2>" + curveName
		if curveNameXX not in gnuplotData:
			gnuplotData[curveNameXX] = ""
			gnuplotData[curveNameYY] = ""

		gnuplotData[curveNameXX] += str(Wi) + "\t" + str(curvePoint['xx']) + "\n"
		gnuplotData[curveNameYY] += str(Wi) + "\t" + str(curvePoint['yy']) + "\n"


firstWi = sorted(data.items())[0][0]
lastWi = sorted(data.items())[-1][0]
WiSteps = 1000

gnuplotTheoryXX = ""
gnuplotTheoryYY = ""
for Wi in numpy.linspace(firstWi, lastWi, WiSteps):
	mu = getLagrangianMultiplicatorRatioFromWeissenbergNumber(Wi)
	predictionXX = 1.0 / mu * (1 + Wi * Wi / (2.0 * mu * mu))
	predictionYY = 1.0 / mu
	gnuplotTheoryXX += str(Wi) + "\t" + str(predictionXX) + "\n"
	gnuplotTheoryYY += str(Wi) + "\t" + str(predictionYY) + "\n"



gnuplot = subprocess.Popen(['gnuplot'], stdin=subprocess.PIPE)
gnuplotCommand = "set terminal wxt persist\n"
gnuplotCommand += "set log x\n"
gnuplotCommand += "set xlabel 'Wi'\n"
gnuplotCommand += "set ylabel '3<Ri^2>/l^2'\n"
gnuplotCommand += "set key outside center bottom horizontal\n"

gnuplotCommand += "plot "
gnuplotCommand += "'-' with lines title 'Theory <x^2>', "
gnuplotCommand += "'-' with lines title 'Theory <y^2>'"
for curveName, _ in gnuplotData.items():
	gnuplotCommand += ", '-' with points title 'Data " + curveName + "'"
gnuplotCommand += "\n"
gnuplotCommand += gnuplotTheoryXX
gnuplotCommand += "e\n"
gnuplotCommand += gnuplotTheoryYY
gnuplotCommand += "e"
for _, curveData in gnuplotData.items():
	gnuplotCommand += "\n" + curveData + "e"

gnuplot.stdin.write(gnuplotCommand.encode("UTF-8"))

exit(0)
