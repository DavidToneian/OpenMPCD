#! /usr/bin/python

import sys

from MPCDAnalysis.DensityProfile import DensityProfile

if len(sys.argv) != 2:
	print("Usage: " + sys.argv[0] + " datafile")
	exit(1)

datafile = sys.argv[1]

densityProfile = DensityProfile(datafile)

densityProfile.plot()
exit(0)
