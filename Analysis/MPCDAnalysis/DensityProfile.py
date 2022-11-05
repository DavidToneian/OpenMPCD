from pylibconfig import Config
import subprocess

class DensityProfile:
	def __init__(self, path):
		config = Config()
		config.readFile(path + "/config.txt")

		self.file = open(path + "/densityProfile.data")
		self.points = {}
		for line in self.file:
			self.parseDataLine(line)

	def plot(self):
		data = self.reduceToXY()

		gnuplotData = ""
		for x, xval in sorted(data.items()):
			for y, yval in sorted(xval.items()):
				gnuplotData += str(x) + "\t" + str(y) + "\t" + str(yval) + "\n"
			gnuplotData += "\n"

		gnuplot = subprocess.Popen(['gnuplot'], stdin=subprocess.PIPE)

		gnuplotCommand = "set terminal wxt persist\n"
		gnuplotCommand += "unset surface\n"
		gnuplotCommand += "set view map\n"
		gnuplotCommand += "set pm3d\n"

		title = self.file.name
		gnuplotCommand += "set title \"" + title + "\" \n"

		gnuplotCommand += "splot '-' using 1:2:3 notitle\n"
		gnuplotCommand += gnuplotData
		gnuplotCommand += "e"

		gnuplot.stdin.write(gnuplotCommand.encode("UTF-8"))

	def parseDataLine(self, line):
		columns = line.split("\t")
		x, y, z = [ float(i) for i in columns[0:3] ]
		density = float(columns[3])

		self.setPoint(x, y, z, density)

	def setPoint(self, x, y, z, density):
		if not x in self.points:
			self.points[x] = {}

		if not y in self.points[x]:
			self.points[x][y] = {}

		self.points[x][y][z] = density

	def reduceToXY(self):
		data = {}

		for x, xval in self.points.items():
			data[x] = {}

			for y, yval in xval.items():
				data[x][y] = 0
				for _, zval in yval.items():
					data[x][y] += zval
				data[x][y] /= len(yval)

		return data
