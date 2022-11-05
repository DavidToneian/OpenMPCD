#!/usr/bin/env python

from __future__ import print_function

import argparse

if __name__ == '__main__':
	programDescription = \
		"Analyses normal mode autocorrelations."
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
		'--saveFilenameZeroTimeDifferenceMiddleModes',
		help = 'Filename, relative to the default directory, to save the ' + \
		       'processed data to, restricting it such that only data for' + \
		       'measurement times of `0` is written, and only for modes ' + \
		       'that are neither `0` nor the maximum node index.',
		type = str, dest = 'saveFilenameZeroTimeDifferenceMiddleModes')
	commandLineParser.add_argument(
		'--metadataFilename',
		help = 'Filename, relative to the default directory, to save ' + \
		       'metadata to. This option is ignored if `--saveFilename` is ' + \
		       'not set.',
		type = str, dest = 'metadataFilename')
	commandLineArguments = commandLineParser.parse_args()


	rundirs = commandLineArguments.rundirs
	if len(rundirs) == 0:
		print('Must specify at least one rundir to save plot files.')
		exit(1)


	from MPCDAnalysis.NormalModeAutocorrelation \
			import NormalModeAutocorrelation

	nma = NormalModeAutocorrelation(rundirs)

	if commandLineArguments.saveFilename is not None:
		with open(commandLineArguments.saveFilename, "w") as f:
			f.write("#")
			f.write("normal-mode-index\t")
			f.write("measurement-time\t")
			f.write("mean-autocorrelation\t")
			f.write("DDDA-optimal-standard-error-of-the-mean\t")
			f.write("sample-size\n")
			for mode in range(0, nma.getNormalModeCount()):
				for T in range(0, nma.getMaximumMeasurementTime() + 1):
					current = nma.getAutocorrelation(mode, T)
					f.write(str(mode) + "\t")
					f.write(str(T) + "\t")
					f.write(str(current.getSampleMean()) + "\t")
					f.write(
						str(current.getOptimalStandardErrorOfTheMean()) \
						+ "\t")
					f.write(str(current.getSampleSize()) + "\t")
					f.write("\n")


	if commandLineArguments.saveFilenameZeroTimeDifferenceMiddleModes \
		is not None:

		filename = \
			commandLineArguments.saveFilenameZeroTimeDifferenceMiddleModes
		with open(filename, "w") as f:
			f.write("#")
			f.write("normal-mode-index\t")
			f.write("mean-autocorrelation\t")
			f.write("DDDA-optimal-standard-error-of-the-mean\t")
			f.write("sample-size\n")
			for mode in range(1, nma.getNormalModeCount() - 1):
				current = nma.getAutocorrelation(mode, 0)
				f.write(str(mode) + "\t")
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
