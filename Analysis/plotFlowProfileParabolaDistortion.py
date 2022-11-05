#! /usr/bin/python

import subprocess
import sys

from MPCDAnalysis.FlowProfile import FlowProfile

if len(sys.argv) < 2:
	print("Usage: " + sys.argv[0] + " [--group groupname] datafiles")
	exit(1)

group = "default"
data = {"default": {} }
arguments = iter(range(1, len(sys.argv)))
for i in arguments:
	if sys.argv[i] == "--group":
		if i + 1 >= len(sys.argv):
			raise RuntimeError("No argument following --group")
		group = sys.argv[i + 1]
		if not group in data:
			data[group] = {}

		arguments.next()
		continue

	flowProfile = FlowProfile(sys.argv[i])
	data[group][flowProfile.getGravity()] = flowProfile.getParabolaQuotientDataVSTheory()

if len(data["default"]) == 0:
	del data["default"]

gnuplotData = ""
for group in data:
	for key in sorted(data[group]):
		gnuplotData += str(key) + "\t" + str(data[group][key]) + "\n"
	gnuplotData += "e\n"

gnuplotCommand = "set terminal wxt persist\n"

gnuplotCommand += "set xlabel 'Gravity'\n"
gnuplotCommand += "set ylabel 'Flow Profile Height Data / Theory'\n"
gnuplotCommand += "plot"
firstGroup = True
for group in data:
	if not firstGroup:
		gnuplotCommand += ", "
	firstGroup = False

	gnuplotCommand += "'-' with linespoints "
	if len(data) == 1 and group == "default":
		gnuplotCommand += "notitle"
	else:
		gnuplotCommand += "title '" + group + "'"

gnuplotCommand += "\n"
gnuplotCommand += gnuplotData

gnuplot = subprocess.Popen(['gnuplot'], stdin=subprocess.PIPE)
gnuplot.stdin.write(gnuplotCommand.encode("UTF-8"))
