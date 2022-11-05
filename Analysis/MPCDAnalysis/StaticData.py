class StaticData:
	def __init__(self, filename):
		self.data = {}

		file = open(filename, 'r')
		for line in file:
			self.readline(line)

	def readline(self, line):
		floatKeys = \
			[
				["dumbbell equilibrium length:", "dumbbellEquilibriumLength"],
				["Weissenberg number:", "weissenbergNumber"],
				["lagrangian multiplier ratio:", "lagrangianMultiplierRatio"]
			]

		for key in floatKeys:
			if self.extractFloat(key[0], key[1], line):
				return

	def extractFloat(self, description, key, line):
		pos = line.find(description)
		if pos == -1:
			return False

		self.data[key] = float(line[pos + len(description) :].strip())
		return True

	def __getitem__(self, val):
		return self.data[val]
