#!/usr/bin/env python

from __future__ import print_function

from MPCDAnalysis.Run import Run

import argparse


if __name__ == '__main__':
	programDescription = \
		"Analyses mean square displacement of logical entities."
	commandLineParser = \
		argparse.ArgumentParser(
			description = programDescription, add_help = True)

	commandLineParser.add_argument(
		'rundirs', help = 'Run directories to analyze', nargs = '*')
	commandLineParser.add_argument(
		'--saveFilename',
		help = 'Filename, relative to the default directory, to save the ' + \
		       'processed data to',
		type = str, dest = 'saveFilename')
	commandLineParser.add_argument(
		'--metadataFilename',
		help = 'Filename, relative to the default directory, to save ' + \
		       'metadata to. This option is ignored if `--saveFilename` is ' + \
		       'not set.',
		type = str, dest = 'metadataFilename')
	commandLineParser.add_argument(
		'--parameterFilename',
		help = 'Filename, relative to the default directory, to save ' + \
		       'simulation parameters to. This option is ignored if ' + \
		       '`--saveFilename` is not set.',
		type = str, dest = 'parameterFilename')
	commandLineArguments = commandLineParser.parse_args()


	rundirs = commandLineArguments.rundirs
	if len(rundirs) == 0:
		print('Must specify at least one rundir to save plot files.')
		exit(1)


	from MPCDAnalysis.LogicalEntityMeanSquareDisplacement \
			import LogicalEntityMeanSquareDisplacement

	msd = LogicalEntityMeanSquareDisplacement(rundirs)

	if commandLineArguments.saveFilename is not None:
		import math
		with open(commandLineArguments.saveFilename, "w") as f:
			f.write("#")
			f.write("measurement-time\t")
			f.write("mean-square-displacement\t")
			f.write("standard-deviation\t")
			f.write("sample-size\n")
			for T in range(1, msd.getMaximumMeasurementTime() + 1):
				current = msd.getMeanSquareDisplacement(T)
				f.write(str(T) + "\t")
				f.write(str(current.getSampleMean()) + "\t")
				f.write(str(math.sqrt(current.getBlockVariance(0))) + "\t")
				f.write(str(current.getSampleSize()) + "\t")
				f.write("\n")

		if commandLineArguments.metadataFilename is not None:
			with open(commandLineArguments.metadataFilename, "w") as f:
				f.write("{")

				f.write('"runDirectories": [' + "\n")
				first = True
				for rundir in rundirs:
					if not first:
						f.write(",")
					f.write('"' + rundir + '"')
					first = False
				f.write("]\n")

				f.write("}")


		if commandLineArguments.parameterFilename is not None:
			numberOfCompletedSweeps = 0
			for rundir in rundirs:
				run = Run(rundir)
				numberOfCompletedSweeps += run.getNumberOfCompletedSweeps()

			additionalParameters = \
				{'numberOfCompletedSweeps': numberOfCompletedSweeps}

			Run(rundirs[0]).getConfiguration().saveToParameterFile(
				commandLineArguments.parameterFilename,
				additionalParameters)

	else:
		print("Linear fit:")

		p1, p2 = msd.fitToData()
		linear = lambda x: p1 * x ** p2

		print(str(p1) + " " + str(p2))

		from MPCDAnalysis.MatplotlibTools.plotAxes import plotAxes
		linear = [[0, 0], [1000, linear(1000)]]
		lines = [linear]
		axes = \
			msd.getMPLAxes(showEstimatedStandardDeviation = True, lines = lines)
		plotAxes(axes)
