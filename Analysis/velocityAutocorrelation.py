#!/usr/bin/env python

from __future__ import print_function

import argparse

if __name__ == '__main__':
	programDescription = \
		"Analysis of center-of-mass velocity autocorrelation."
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
		'--minimumTime',
		help =
			'Discard all measurements involving simulation times smaller ' + \
			'than this value.',
		type = float,
		dest = 'minimumTime',
		default = 0.0)
	commandLineArguments = commandLineParser.parse_args()


	rundirs = commandLineArguments.rundirs
	if len(rundirs) == 0:
		print('Must specify at least one rundir.')
		exit(1)


	from MPCDAnalysis.VelocityAutocorrelation import VelocityAutocorrelation

	vac = VelocityAutocorrelation(rundirs, commandLineArguments.minimumTime)

	if commandLineArguments.saveFilename is not None:
		with open(commandLineArguments.saveFilename, "w") as f:
			f.write("#")
			f.write("correlation-time\t")
			f.write("mean-autocorrelation\t")
			f.write("DDDA-optimal-standard-error-of-the-mean\t")
			f.write("sample-size\n")
			for t in vac.getCorrelationTimes():
				current = vac.getAutocorrelation(t)
				f.write(str(t) + "\t")
				f.write(str(current.getSampleMean()) + "\t")
				f.write(
					str(current.getOptimalStandardErrorOfTheMean()) \
					+ "\t")
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
