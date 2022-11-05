from .Utilities import readValuePairsFromFile
from collections import deque, OrderedDict
import math
import subprocess

class Data2D:
	def __init__(self, filename=None, minXValue=None):
		if filename is None:
			self.data = OrderedDict()
		else:
			self.readFromFile(filename, minXValue)

		self.errors = OrderedDict()

	def readFromFile(self, filename, minXValue):
		self.data = readValuePairsFromFile(filename, minXValue)

	def addPoint(self, point, value, error=None):
			self.data[point] = value
			if error is not None:
				self.errors[point] = error

	def sort(self):
		self.data = OrderedDict(sorted(self.data.items()))
		self.errors = OrderedDict(sorted(self.errors.items()))

	def getNearestPoint(self, x):
		self.sort()

		iterator = iter(self.data.items())
		firstX, firstY = next(iterator)

		if x <= firstX:
			return (firstX, firstY)

		lastX = firstX
		lastY = firstY
		lastDistance = x - firstX

		for currentX, currentY in iterator:
			newDistance = x - currentX

			if newDistance < 0:
				if abs(newDistance) < lastDistance:
					return (currentX, currentY)
				return (lastX, lastY)

			lastX = currentX
			lastY = currentY
			lastDistance = x - currentX

		return (lastX, lastY)

	def plot(self, plotEveryNthPoint=1):
		gnuplotData = ""
		counter = 0
		for x, y in sorted(self.data.items()):
			if counter % plotEveryNthPoint == 0:
				gnuplotData += str(x) + "\t" + str(y) + "\n"

			counter += 1

		gnuplot = subprocess.Popen(['gnuplot'], stdin=subprocess.PIPE)

		gnuplotCommand = "set terminal wxt persist\n"
		gnuplotCommand += "plot '-' with linespoints notitle\n"
		gnuplotCommand += gnuplotData
		gnuplotCommand += "e"

		gnuplot.stdin.write(gnuplotCommand.encode("UTF-8"))

	def getArithmeticMean(self):
		sum_ = 0
		count = 0
		for _, y in self.data.items():
			count += 1
			sum_ += y

		return sum_ / count

	def getRootMeanSquaredDeviationFromArithmeticMean(self):
		mean = self.getArithmeticMean()

		sum_ = 0
		count = 0
		for _, y in self.data.items():
			count += 1
			sum_ += (y - mean) ** 2

		return math.sqrt(sum_ / count)

	def getSimpleMovingAverageDict(self, windowsize):
		self.sort()

		window = deque()
		averages = OrderedDict()

		for x, y in self.data.items():
			window.append((x, y))

			firstX = window[0][0]
			if x - firstX >= windowsize:
				sum_ = 0
				count = 0
				for value in window:
					sum_ += value[1]
					count += 1

				averages.update({x: sum_ / count})

				window.popleft()

		return averages

	def getAverageFromSimpleMovingAverage(self, windowsize):
		averages = self.getSimpleMovingAverageDict(windowsize)

		sum_ = 0
		count = 0
		for _, y in averages.items():
			sum_ += y
			count += 1

		return sum_ / count

	def getLocalExtremaByComparisonFunction(self, comparisonFunction, includeBoundaries):
		self.sort()

		extrema = {}

		items = {key: value for key, value in enumerate(self.data.items())}
		print(items)
		for index in items:
			value = items[index]
			x = value[0]
			y = value[1]

			if index == 0:
				if not includeBoundaries:
					continue

				if index + 1 not in items:
					extrema[x] = y
					print(items)
					print("break1")
					break

				if comparisonFunction(y, items[index + 1][1]):
					extrema[x] = y

				continue

			if index + 1 not in items:
				if not includeBoundaries:
					continue

				if comparisonFunction(y, items[index - 1][1]):
					extrema[x] = y

				continue

			if not comparisonFunction(y, items[index - 1][1]):
				continue
			if not comparisonFunction(y, items[index + 1][1]):
				continue
			extrema[x] = y

		return extrema


	def getLocalMaxima(self, includeBoundaries=True):
		greaterThan = lambda x, y: x > y
		return self.getLocalExtremaByComparisonFunction(greaterThan, includeBoundaries)

	def getLocalMinima(self, includeBoundaries=True):
		lessThan = lambda x, y: x < y
		return self.getLocalExtremaByComparisonFunction(lessThan, includeBoundaries)

	def getLocalExtrema(self, includeBoundaries=True):
		minima = self.getLocalMinima(includeBoundaries)
		maxima = self.getLocalMaxima(includeBoundaries)
		return OrderedDict(sorted(minima.items() + maxima.items()))


	def getData(self, sortFirst=True):
		if sortFirst:
			self.sort()

		return self.data

	def getKeysAndValues(self, sortFirst=True):
		if sortFirst:
			self.sort()

		return zip(*self.data.items())

	def getKeysAndValuesAndErrors(self, sortFirst=True):
		if sortFirst:
			self.sort()

		keys = []
		values = []
		errors = []
		for key, value in self.data.items():
			keys.append(key)
			values.append(value)
			if key in self.errors:
				error = self.errors[key]
			else:
				error = None
			errors.append(error)

		return [keys, values, errors]

	def getSize(self):
		return len(self.data)

	def save(self, filename, sortFirst=True):
		"""
		Saves the data to the given filename.

		@param[in] filename  The file path to save to.
		@param[in] sortFirst Whether to sort the points first, so that their x coordinate is ascending.
		"""

		self.writeTo(open(filename, 'w'), sortFirst)

	def writeTo(self, stream, sortFirst=True):
		"""
		Writes the data to the given object.

		@param[in] stream    The object to write to.
		@param[in] sortFirst Whether to sort the points first, so that their x coordinate is ascending.
		"""

		if sortFirst:
			self.sort()

		for x, y in self.data.items():
			stream.write(str(x) + "\t" + str(y))
			if x in self.errors:
				stream.write("\t" + str(self.errors[x]))
			stream.write("\n")
