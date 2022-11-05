#! /usr/bin/python

import sys

from MPCDAnalysis.FlowProfile import FlowProfile

if len(sys.argv) < 2:
	print("Usage: " + sys.argv[0] + " datafile [datafile ...]")
	exit(1)

datafile = sys.argv[1]

flowProfile = FlowProfile(datafile)

for i in range(2, len(sys.argv)):
	flowProfile.addDataFile(sys.argv[i])

flowProfile.plotPoiseuilleFlowProfileX()
exit(0)
