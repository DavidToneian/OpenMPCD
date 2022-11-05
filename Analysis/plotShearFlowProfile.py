#! /usr/bin/python

import argparse

from MPCDAnalysis.FlowProfile import FlowProfile

programDescription = 'Plots a shear flow profile.'
commandLineParser = argparse.ArgumentParser(
	description = programDescription, add_help = True)
commandLineParser.add_argument(
	'rundirs', help = 'Run directories to analyze', nargs = '*')
commandLineArguments = commandLineParser.parse_args()

if len(commandLineArguments.rundirs) < 1:
	print("Too few datafiles")
	exit(1)


flowProfile = FlowProfile(commandLineArguments.rundirs[0])
for i in range(1, len(commandLineArguments.rundirs)):
	flowProfile.addRun(commandLineArguments.rundirs[i])

from MPCDAnalysis.MatplotlibTools.plotAxes import plotAxes
plotAxes(flowProfile.getMPLAxesForFlowAlongXAsFunctionOfY())

exit(0)
