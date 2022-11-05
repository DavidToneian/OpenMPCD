from .Utilities import readValuePairsFromFile

class EmpiricalDistribution:
	def __init__(self, filename):
		self.readFromFile(filename)

	def readFromFile(self, filename):
		self.data = readValuePairsFromFile(filename)

	def getSampleMean(self):
		sum_ = 0
		count = 0
		for x, y in self.data.items():
			count += y
			sum_ += x * y

		return sum_ / count
