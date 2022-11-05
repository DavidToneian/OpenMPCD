#! /usr/bin/python

"""
Plots the given files, flipping the sign of negative y-values and showing them as part of a dashed line.
"""

import MPCDAnalysis.PlotTools as PlotTools

import argparse
import matplotlib.pyplot as plt

programDescription = __doc__
commandLineParser = argparse.ArgumentParser(description=programDescription, add_help=True)
commandLineParser.add_argument('filenames', help='The files to plot', nargs='+')
commandLineParser.add_argument(
	'--log-x', help='plot x axis logarithmically',
	action='store_true', dest='logx')
commandLineParser.add_argument(
	'--log-y', help='plot y axis logarithmically',
	action='store_true', dest='logy')
commandLineArguments = commandLineParser.parse_args()




fig = plt.figure()
ax = plt.subplot(1, 1, 1)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])




if commandLineArguments.logx:
	plt.xscale("log")
if commandLineArguments.logy:
	plt.yscale("log")




for filename in commandLineArguments.filenames:
	with open(filename, 'r') as file:
		plotX = []
		plotY = []
		for line in file:
			if line[0] == '#':
				continue

			x, y = [float(val) for val in line.split()]

			if x == 0 or x > 1e3:
				continue

			plotX.append(x)
			plotY.append(y)

		if '1' in filename:
			etaMPC = '8'
		elif '2' in filename:
			etaMPC = '16'
		else:
			etaMPC = '4'

		PlotTools.plotDashedIfNegative(plotX, plotY, logY=commandLineArguments.logy, label=r'$\eta_{MPC} = ' + etaMPC + '$')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
