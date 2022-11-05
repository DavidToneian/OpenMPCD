#! /usr/bin/python

import MPCDAnalysis.Utilities as Utilities

import argparse
import matplotlib.pyplot

deltaTPrecision = 8

def processRundir(rundir, data):
	filename = rundir + '/normalModeAutocorrelations.data'
	with Utilities.openPossiblyCompressedFile(filename) as file:
		for line in file:
			t0, t, normalCoordinateIndex, correlation = [float(x) for x in line.split()]
			deltaT = round(t - t0, deltaTPrecision)

			if normalCoordinateIndex not in data:
				data[normalCoordinateIndex] = {}
			if deltaT not in data[normalCoordinateIndex]:
				data[normalCoordinateIndex][deltaT] = []

			data[normalCoordinateIndex][deltaT].append(correlation)


if __name__ == '__main__':
	programDescription = 'Computes the polymer relaxation times.'
	commandLineParser = argparse.ArgumentParser(description=programDescription, add_help=True)
	commandLineParser.add_argument('rundirs', help='Run directories to analyze', nargs='*')
	commandLineParser.add_argument(
		'--modes',
		help=r'Normal modes to include, as a comma-separated list of ' + \
		     r'integers, defaults to all modes',
		type=str, dest='modes')
	commandLineParser.add_argument(
		'--saveFilename',
		help='Filename to save the processed data to. ' +
		     'If not given, plot data instead.',
		type=str,
		dest='saveFilename')
	commandLineArguments = commandLineParser.parse_args()

	if len(commandLineArguments.rundirs) < 1:
		print("Too few rundirs")
		exit(1)

	data = {}
	for rundir in commandLineArguments.rundirs:
		processRundir(rundir, data)

	modesToInclude = None
	if commandLineArguments.modes:
		modesToInclude = [int(x) for x in commandLineArguments.modes.split(',')]

	results = {}
	for normalCoordinateIndex, normalCoordinateData in data.items():
		if modesToInclude is not None:
			if normalCoordinateIndex not in modesToInclude:
				continue

		results[normalCoordinateIndex] = {}
		first = None
		for deltaT, deltaTData in normalCoordinateData.items():
			value = sum(deltaTData) / len(deltaTData)
			if first is None:
				first = sum(deltaTData) / len(deltaTData)

			results[normalCoordinateIndex][deltaT] = value / first

	if commandLineArguments.saveFilename is None:
		matplotlib.pyplot.yscale('log')
		for normalCoordinateIndex, normalCoordinateData in results.items():
			x = sorted(normalCoordinateData.keys())
			y = [normalCoordinateData[key] for key in x]
			label = "i = " + str(normalCoordinateIndex)
			matplotlib.pyplot.plot(x, y, label=label)

		matplotlib.pyplot.legend()
		matplotlib.pyplot.show()
	else:
		for normalCoordinateIndex, normalCoordinateData in results.items():
			datafilePath = commandLineArguments.saveFilename + '--normalCoordinateIndex=' + str(int(normalCoordinateIndex))
			with open(datafilePath, 'w') as datafile:
				datafile.write('#rundirs = [')
				_isFirstRundir = True
				for _rundir in commandLineArguments.rundirs:
					if not _isFirstRundir:
						datafile.write(', ')

					_isFirstRundir = False

					datafile.write("'")
					datafile.write(_rundir)
					datafile.write("'")

				datafile.write(']\n')

				for x in sorted(normalCoordinateData.keys()):
					datafile.write(str(x) + '\t' + str(normalCoordinateData[x]) + '\n')
