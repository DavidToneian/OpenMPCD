#! /usr/bin/python

import sys

from MPCDAnalysis.EmpiricalDistribution import EmpiricalDistribution

if len(sys.argv) != 2:
	print("Usage: " + sys.argv[0] + " datafile")
	exit(1)

data = EmpiricalDistribution(sys.argv[1])
print(data.getSampleMean())
