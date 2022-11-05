#! /usr/bin/python

import sys

from MPCDAnalysis.Data2D import Data2D

if len(sys.argv) not in [2, 3]:
	print("Usage: " + sys.argv[0] + " [--plotEveryNthPoint=value] datafile")
	exit(1)

plotEveryNthPoint = 1
dataPath = sys.argv[1]

if len(sys.argv) == 3 and sys.argv[1].startswith("--plotEveryNthPoint="):
	plotEveryNthPoint = int(sys.argv[1][len("--plotEveryNthPoint="):])
	dataPath = sys.argv[2]

data = Data2D(dataPath)

data.plot(plotEveryNthPoint)
