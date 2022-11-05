from __future__ import division

import pytest

from MPCDAnalysis.OnTheFlyStatisticsDDDA import OnTheFlyStatisticsDDDA

def test_emptyInstance():
	ddda = OnTheFlyStatisticsDDDA()

	assert ddda.getSampleSize() == 0

	with pytest.raises(Exception):
		ddda.getSampleMean()

	with pytest.raises(Exception):
		ddda.getMaximumBlockSize()

	assert ddda.getMaximumBlockID() == 0


def test_oneDatum():
	ddda = OnTheFlyStatisticsDDDA()

	ddda.addDatum(5)

	assert ddda.getSampleSize() == 1
	assert ddda.getSampleMean() == 5
	assert ddda.getMaximumBlockSize() == 1
	assert ddda.getMaximumBlockID() == 0


def test_twoData_int():
	ddda = OnTheFlyStatisticsDDDA()

	data = [5, 2]

	for datum in data:
		ddda.addDatum(datum)

	assert ddda.getSampleSize() == 2
	assert ddda.getSampleMean() == sum(data) / len(data)
	assert ddda.getMaximumBlockSize() == 2
	assert ddda.getMaximumBlockID() == 1


def test_twoData_float():
	ddda = OnTheFlyStatisticsDDDA()

	data = [5, 2.0]

	for datum in data:
		ddda.addDatum(datum)

	assert ddda.getSampleSize() == 2
	assert ddda.getSampleMean() == sum(data) / len(data)
	assert ddda.getMaximumBlockSize() == 2
	assert ddda.getMaximumBlockID() == 1


def test_threeData():
	ddda = OnTheFlyStatisticsDDDA()

	data = [5, 2.0, -1.2]

	for datum in data:
		ddda.addDatum(datum)

	assert ddda.getSampleSize() == 3
	assert ddda.getSampleMean() == sum(data) / len(data)
	assert ddda.getMaximumBlockSize() == 2
	assert ddda.getMaximumBlockID() == 1


def test_fourData():
	ddda = OnTheFlyStatisticsDDDA()

	data = [5, 2.0, -1.2, 40.0]

	for datum in data:
		ddda.addDatum(datum)

	from pytest import approx

	assert ddda.getSampleSize() == 4
	assert ddda.getSampleMean() == approx(sum(data) / len(data))
	assert ddda.getMaximumBlockSize() == 4
	assert ddda.getMaximumBlockID() == 2


def test_dynamicData():
	dataSizes = \
		[
			1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		]

	for dataSize in dataSizes:
		import math
		import random
		data = [random.random() for i in range(0, dataSize)]

		ddda = OnTheFlyStatisticsDDDA()
		for datum in data:
			ddda.addDatum(datum)

		blockSizeCount = 1
		maximumBlockSize = 1
		while True:
			if maximumBlockSize * 2 > dataSize:
				break
			blockSizeCount += 1
			maximumBlockSize *= 2

		assert ddda.getSampleSize() == len(data)
		assert ddda.getSampleMean() == pytest.approx(sum(data) / len(data))
		assert ddda.getMaximumBlockSize() == maximumBlockSize
		assert ddda.getMaximumBlockID() == blockSizeCount - 1


def test_merge():
	dataSizes = \
		[
			1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100,
			1000,
		]

	ddda = OnTheFlyStatisticsDDDA()
	with pytest.raises(TypeError):
		ddda.merge(None)
	with pytest.raises(TypeError):
		ddda.merge([ddda])
	with pytest.raises(TypeError):
		from MPCDAnalysis.OnTheFlyStatistics import OnTheFlyStatistics
		ddda.merge(OnTheFlyStatistics())


	def approximatelyEquivalent(lhs, rhs):
		assert lhs is not rhs

		if lhs.getSampleSize() != rhs.getSampleSize():
			return False
		if lhs.getSampleMean() != pytest.approx(rhs.getSampleMean()):
			return False
		if lhs.getMaximumBlockSize() != rhs.getMaximumBlockSize():
			return False
		if lhs.getMaximumBlockID() != rhs.getMaximumBlockID():
			return False
		for i in range(0, lhs.getMaximumBlockID() + 1):
			if lhs.hasBlockVariance(i) != rhs.hasBlockVariance(i):
				return False
			if not lhs.hasBlockVariance(i):
				continue

			blockVarianceLHS = lhs.getBlockVariance(i)
			blockVarianceRHS = rhs.getBlockVariance(i)

			if i == 0:
				if blockVarianceLHS != pytest.approx(blockVarianceRHS):
					return False
			else:
				# For blocked data, the way in which the data are blocked may
				# differ in the arguments provided to this function, and in case
				# some blocks are not complete, the data points that are waiting
				# may differ between the arguments.
				# A meaningful criterion is therefore hard to formulate.
				pass

		return True

	import random
	for dataSize1 in dataSizes:
		for dataSize2 in dataSizes:
			data1 = [random.random() for _ in range(0, dataSize1)]
			data2 = [random.random() for _ in range(0, dataSize2)]

			ddda1 = OnTheFlyStatisticsDDDA()
			ddda2 = OnTheFlyStatisticsDDDA()

			ddda11 = OnTheFlyStatisticsDDDA()
			ddda12 = OnTheFlyStatisticsDDDA()

			for datum in data1:
				ddda1.addDatum(datum)
				ddda11.addDatum(datum)
				ddda12.addDatum(datum)

			for datum in data1:
				ddda11.addDatum(datum)

			for datum in data2:
				ddda2.addDatum(datum)
				ddda12.addDatum(datum)

			import copy
			copied = copy.deepcopy(ddda1)
			assert ddda1 == copied
			assert not ddda1 is copied

			assert not approximatelyEquivalent(ddda1, ddda12)
			ddda1.merge(ddda2)
			assert approximatelyEquivalent(ddda1, ddda12)

			#merge self:
			assert not approximatelyEquivalent(copied, ddda11)
			copied.merge(copied)
			assert approximatelyEquivalent(copied, ddda11)



def test_hasBlockVariance_getBlockVariance():
	dataSizes = \
		[
			0, 1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		]

	for dataSize in dataSizes:
		import random
		data = [random.random() for i in range(0, dataSize)]

		ddda = OnTheFlyStatisticsDDDA()
		for datum in data:
			ddda.addDatum(datum)

		with pytest.raises(TypeError):
			ddda.hasBlockVariance(0.0)

		with pytest.raises(TypeError):
			ddda.getBlockVariance(0.0)

		with pytest.raises(ValueError):
			ddda.getBlockVariance(-1)

		with pytest.raises(ValueError):
			ddda.hasBlockVariance(-1)

		with pytest.raises(ValueError):
			ddda.hasBlockVariance(ddda.getMaximumBlockID() + 1)

		with pytest.raises(ValueError):
			ddda.getBlockVariance(ddda.getMaximumBlockID() + 1)


		for blockID in range(0, ddda.getMaximumBlockID() + 1):
			blockSize = 2 ** blockID

			if dataSize / blockSize < 2:
				assert not ddda.hasBlockVariance(blockID)
				with pytest.raises(RuntimeError):
					ddda.getBlockVariance(blockID)
			else:
				from MPCDAnalysis.OnTheFlyStatistics import OnTheFlyStatistics
				stat = OnTheFlyStatistics()
				tmp = OnTheFlyStatistics()
				for datum in data:
					tmp.addDatum(datum)

					if tmp.getSampleSize() == blockSize:
						stat.addDatum(tmp.getSampleMean())
						tmp = OnTheFlyStatistics()

				expected = stat.getSampleVariance()

				assert ddda.hasBlockVariance(blockID)
				assert ddda.getBlockVariance(blockID) == pytest.approx(expected)


def test_getBlockStandardDeviation():
	dataSizes = \
		[
			0, 1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		]

	for dataSize in dataSizes:
		import random
		data = [random.random() for i in range(0, dataSize)]

		ddda = OnTheFlyStatisticsDDDA()
		for datum in data:
			ddda.addDatum(datum)

		with pytest.raises(TypeError):
			ddda.getBlockStandardDeviation(0.0)

		with pytest.raises(ValueError):
			ddda.getBlockStandardDeviation(-1)

		with pytest.raises(ValueError):
			ddda.getBlockStandardDeviation(ddda.getMaximumBlockID() + 1)


		for blockID in range(0, ddda.getMaximumBlockID() + 1):
			if not ddda.hasBlockVariance(blockID):
				with pytest.raises(RuntimeError):
					ddda.getBlockStandardDeviation(blockID)

				continue

			import math
			expected = math.sqrt(ddda.getBlockVariance(blockID))
			assert ddda.getBlockStandardDeviation(blockID) == expected


def test_getSampleStandardDeviation():
	dataSizes = \
		[
			0, 1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		]

	for dataSize in dataSizes:
		import random
		data = [random.random() for i in range(0, dataSize)]

		ddda = OnTheFlyStatisticsDDDA()
		for datum in data:
			ddda.addDatum(datum)

		if not ddda.hasBlockVariance(0):
			with pytest.raises(RuntimeError):
				ddda.getSampleStandardDeviation()

			continue

		expected = ddda.getBlockStandardDeviation(0)
		assert ddda.getSampleStandardDeviation() == expected



def test_getBlockStandardErrorOfTheMean():
	dataSizes = \
		[
			0, 1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		]

	for dataSize in dataSizes:
		import random
		data = [random.random() for i in range(0, dataSize)]

		ddda = OnTheFlyStatisticsDDDA()
		for datum in data:
			ddda.addDatum(datum)


		with pytest.raises(TypeError):
			ddda.getBlockStandardErrorOfTheMean(0.0)

		with pytest.raises(ValueError):
			ddda.getBlockStandardErrorOfTheMean(-1)

		with pytest.raises(ValueError):
			ddda.getBlockStandardErrorOfTheMean(ddda.getMaximumBlockID() + 1)


		for blockID in range(0, ddda.getMaximumBlockID() + 1):
			blockSize = 2 ** blockID

			if ddda.hasBlockVariance(blockID):
				from MPCDAnalysis.OnTheFlyStatistics import OnTheFlyStatistics
				stat = OnTheFlyStatistics()
				tmp = OnTheFlyStatistics()
				for datum in data:
					tmp.addDatum(datum)

					if tmp.getSampleSize() == blockSize:
						stat.addDatum(tmp.getSampleMean())
						tmp = OnTheFlyStatistics()

				result = ddda.getBlockStandardErrorOfTheMean(blockID)
				expected = stat.getStandardErrorOfTheMean()

				assert result == pytest.approx(expected)
			else:
				with pytest.raises(RuntimeError):
					ddda.getBlockStandardErrorOfTheMean(blockID)


def test_getEstimatedStandardDeviationOfBlockStandardErrorOfTheMean():
	dataSizes = \
		[
			0, 1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		]

	for dataSize in dataSizes:
		import random
		data = [random.random() for i in range(0, dataSize)]

		ddda = OnTheFlyStatisticsDDDA()

		DDDA = OnTheFlyStatisticsDDDA
		f = DDDA.getEstimatedStandardDeviationOfBlockStandardErrorOfTheMean

		for datum in data:
			ddda.addDatum(datum)


		with pytest.raises(TypeError):
			ddda.getEstimatedStandardDeviationOfBlockStandardErrorOfTheMean(0.0)

		with pytest.raises(ValueError):
			ddda.getEstimatedStandardDeviationOfBlockStandardErrorOfTheMean(-1)

		with pytest.raises(ValueError):
			ddda.getEstimatedStandardDeviationOfBlockStandardErrorOfTheMean(
				ddda.getMaximumBlockID() + 1)


		for blockID in range(0, ddda.getMaximumBlockID() + 1):
			if not ddda.hasBlockVariance(blockID):
				with pytest.raises(RuntimeError):
					f(ddda, blockID)
				continue

			esd = f(ddda, blockID)
			sem = ddda.getBlockStandardErrorOfTheMean(blockID)

			blockSize = 2 ** blockID
			reducedSampleSize = dataSize // blockSize

			import math
			assert esd == sem / math.sqrt(2 * reducedSampleSize)


def test_optimal_standard_error():
	from MPCDAnalysis.OnTheFlyStatistics import OnTheFlyStatistics

	dataSizes = \
		[
			0, 1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		]

	for dataSize in dataSizes:
		import random
		data = [random.random() for i in range(0, dataSize)]

		ddda = OnTheFlyStatisticsDDDA()
		stat = OnTheFlyStatistics()
		for datum in data:
			ddda.addDatum(datum)
			stat.addDatum(datum)


		if dataSize < 2:
			with pytest.raises(RuntimeError):
				ddda.getOptimalBlockIDForStandardErrorOfTheMean()
			with pytest.raises(RuntimeError):
				ddda.optimalStandardErrorOfTheMeanEstimateIsReliable()
			with pytest.raises(RuntimeError):
				ddda.getOptimalStandardErrorOfTheMean()

			continue



		optimalBlockID = ddda.getOptimalBlockIDForStandardErrorOfTheMean()
		optimalBlockSize = 2 ** optimalBlockID

		rawSE = stat.getStandardErrorOfTheMean()

		blocks = []
		for block in range(0, ddda.getMaximumBlockID()):
			blockSize = 2 ** block

			blockStat = OnTheFlyStatistics()
			tmp = OnTheFlyStatistics()
			for datum in data:
				tmp.addDatum(datum)

				if tmp.getSampleSize() == blockSize:
					blockStat.addDatum(tmp.getSampleMean())
					tmp = OnTheFlyStatistics()
			blocks.append(blockStat)

		expectedSE = blocks[optimalBlockID].getStandardErrorOfTheMean()

		_resultSE = ddda.getOptimalStandardErrorOfTheMean()
		assert _resultSE == pytest.approx(expectedSE)


		if ddda.optimalStandardErrorOfTheMeanEstimateIsReliable():
			assert optimalBlockSize < dataSize / 50.0
		else:
			assert optimalBlockSize >= dataSize / 50.0

		criteria = []
		for blockID in range(0, ddda.getMaximumBlockID() + 1):
			if not ddda.hasBlockVariance(blockID):
				criteria.append(False)
				continue

			currentSE = blocks[blockID].getStandardErrorOfTheMean()
			quotient = currentSE / rawSE
			criterion = 2 ** (blockID * 3) > 2 * dataSize * quotient ** 4

			criteria.append(criterion)

		assert len(criteria) == ddda.getMaximumBlockID() + 1
		for blockID, criterion in enumerate(criteria):
			if blockID < optimalBlockID - 1:
				continue

			if blockID == optimalBlockID - 1:
				assert criterion == False
				continue

			if blockID == ddda.getMaximumBlockID() - 1:
				if not ddda.hasBlockVariance(ddda.getMaximumBlockID()):
					if optimalBlockID == blockID:
						continue

			if blockID == ddda.getMaximumBlockID():
				if optimalBlockID == ddda.getMaximumBlockID():
					continue
				else:
					if not ddda.hasBlockVariance(ddda.getMaximumBlockID()):
						continue

			assert criterion == True


def test_getOptimalBlockIDForStandardErrorOfTheMean_zero_variance():
	dataSizes = \
		[
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		]

	for dataSize in dataSizes:
		import random
		datum = random.random()
		data = [datum for i in range(0, dataSize)]

		ddda = OnTheFlyStatisticsDDDA()
		for datum in data:
			ddda.addDatum(datum)

		assert ddda.getOptimalBlockIDForStandardErrorOfTheMean() == 0


def test_serializeToString():
	dataSizes = \
		[
			0, 1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		]

	for dataSize in dataSizes:
		import random
		data = [random.random() for _ in range(0, dataSize)]

		stat = OnTheFlyStatisticsDDDA()
		assert isinstance(stat.serializeToString(), str)
		for datum in data:
			stat.addDatum(datum)
			assert isinstance(stat.serializeToString(), str)


def test_unserializeFromString():
	stat = OnTheFlyStatisticsDDDA()

	with pytest.raises(TypeError):
		stat.unserializeFromString(["foo"])
	with pytest.raises(TypeError):
		stat.unserializeFromString(stat)

	with pytest.raises(ValueError):
		stat.unserializeFromString("")
	with pytest.raises(ValueError):
		stat.unserializeFromString("foo")
	with pytest.raises(ValueError):
		stat.unserializeFromString("123|0")
	with pytest.raises(ValueError):
		stat.unserializeFromString("1;0;0;0")
	with pytest.raises(ValueError):
		stat.unserializeFromString("1|1|1;0;0;1|")
	with pytest.raises(ValueError):
		stat.unserializeFromString("1|-1")


	stat.unserializeFromString("1|0")
	assert stat.getSampleSize() == 0

	stat = OnTheFlyStatisticsDDDA()
	unserialized = OnTheFlyStatisticsDDDA()
	unserialized.unserializeFromString(stat.serializeToString())
	assert unserialized == stat
	unserialized.unserializeFromString(stat.serializeToString())
	assert unserialized == stat


	def approximatelyEqual(lhs, rhs):
		if lhs.getSampleSize() != rhs.getSampleSize():
			return False

		if lhs.getSampleSize() == 0:
			return True

		if lhs.getSampleMean() != pytest.approx(rhs.getSampleMean()):
			return False

		if lhs.getMaximumBlockID() != rhs.getMaximumBlockID():
			return False

		for i in range(0, lhs.getMaximumBlockID() + 1):
			if lhs.hasBlockVariance(i) != rhs.hasBlockVariance(i):
				return False
			if not lhs.hasBlockVariance(i):
				continue

			expected = pytest.approx(rhs.getBlockVariance(i))
			if lhs.getBlockVariance(i) != expected:
				return False

		return True


	import random
	for _ in range(0, 50):
		for _ in range(0, random.randint(1, 5)):
			stat.addDatum(random.random())

		assert not approximatelyEqual(unserialized, stat)
		unserialized.unserializeFromString(stat.serializeToString())
		assert approximatelyEqual(unserialized, stat)
		unserialized.unserializeFromString(stat.serializeToString())
		assert approximatelyEqual(unserialized, stat)

		import copy
		statCopy = copy.deepcopy(stat)
		unserializedCopy = copy.deepcopy(unserialized)
		for _ in range(0, 50):
			#test that the waiting samples are (un)serialized correctly
			datum = random.random()
			statCopy.addDatum(datum)
			unserializedCopy.addDatum(datum)
			assert approximatelyEqual(unserializedCopy, statCopy)


	stat = OnTheFlyStatisticsDDDA()
	stat.addDatum(1)
	stat.addDatum(2)
	stat.addDatum(3)
	stat.addDatum(4)
	stat.addDatum(5)

	from MPCDAnalysis.OnTheFlyStatistics import OnTheFlyStatistics
	block1 = OnTheFlyStatistics()
	block2 = OnTheFlyStatistics()
	block3 = OnTheFlyStatistics()
	block1.addDatum(1)
	block1.addDatum(2)
	block1.addDatum(3)
	block1.addDatum(4)
	block1.addDatum(5)
	block2.addDatum((1 + 2) / 2.0)
	block2.addDatum((3 + 4) / 2.0)
	block3.addDatum(((1 + 2) / 2.0) + ((3 + 4) / 2.0))

	myState = "1|" + "3"
	myState += "|" + block1.serializeToString()
	myState += "|" + block2.serializeToString()
	myState += "|" + block3.serializeToString()
	myState += "|" + str(5)
	myState += "|"
	myState += "|"

	unserialized = OnTheFlyStatisticsDDDA()
	unserialized.unserializeFromString(myState)
	assert approximatelyEqual(unserialized, stat)


def test_getMPLAxes():
	dataSizes = \
		[
			0, 1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		]

	for dataSize in dataSizes:
		import random
		data = [random.random() for _ in range(0, dataSize)]

		ddda = OnTheFlyStatisticsDDDA()
		for datum in data:
			ddda.addDatum(datum)

		with pytest.raises(TypeError):
			ddda.getMPLAxes(0)
		with pytest.raises(TypeError):
			ddda.getMPLAxes(1)

		import matplotlib.axes
		assert isinstance(ddda.getMPLAxes(False), matplotlib.axes.Axes)
		assert isinstance(ddda.getMPLAxes(True), matplotlib.axes.Axes)


def test___eq_____ne__():
	dataSizes = \
		[
			0, 1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		]


	ddda = OnTheFlyStatisticsDDDA()
	with pytest.raises(TypeError):
		ddda.__eq__(None)
	with pytest.raises(TypeError):
		ddda.__eq__([ddda])
	with pytest.raises(TypeError):
		from MPCDAnalysis.OnTheFlyStatistics import OnTheFlyStatistics
		ddda.__eq__(OnTheFlyStatistics())

	with pytest.raises(TypeError):
		ddda == None
	with pytest.raises(TypeError):
		ddda == [ddda]
	with pytest.raises(TypeError):
		from MPCDAnalysis.OnTheFlyStatistics import OnTheFlyStatistics
		ddda == OnTheFlyStatistics()

	with pytest.raises(TypeError):
		ddda.__ne__(None)
	with pytest.raises(TypeError):
		ddda.__ne__([ddda])
	with pytest.raises(TypeError):
		from MPCDAnalysis.OnTheFlyStatistics import OnTheFlyStatistics
		ddda.__ne__(OnTheFlyStatistics())

	with pytest.raises(TypeError):
		ddda != None
	with pytest.raises(TypeError):
		ddda != [ddda]
	with pytest.raises(TypeError):
		from MPCDAnalysis.OnTheFlyStatistics import OnTheFlyStatistics
		ddda != OnTheFlyStatistics()


	for dataSize in dataSizes:
		import random

		original = OnTheFlyStatisticsDDDA()
		same = OnTheFlyStatisticsDDDA()
		missingBeginning = OnTheFlyStatisticsDDDA()
		missingEnd = OnTheFlyStatisticsDDDA()
		modifiedBeginning = OnTheFlyStatisticsDDDA()
		modifiedEnd = OnTheFlyStatisticsDDDA()
		for i in range(0, dataSize):
			datum = random.random()

			original.addDatum(datum)
			same.addDatum(datum)

			if i == 0:
				modifiedBeginning.addDatum(datum + 1)
			else:
				missingBeginning.addDatum(datum)
				modifiedBeginning.addDatum(datum)

			if i == dataSize - 1:
				modifiedEnd.addDatum(datum - 1)
			else:
				missingEnd.addDatum(datum)
				modifiedEnd.addDatum(datum)

		import copy
		copied = copy.deepcopy(original)
		instances = \
			[
				original, same, copied,
				missingBeginning, missingEnd,
				modifiedBeginning, modifiedEnd
			]

		for ddda in instances:
			assert ddda == ddda
			assert ddda.__eq__(ddda)
			assert not ddda != ddda
			assert not ddda.__ne__(ddda)

		for ddda1 in instances:
			for ddda2 in instances:
				shouldCompareEqual1 = False
				if ddda1 is original:
					shouldCompareEqual1 = True
				elif ddda1 is same:
					shouldCompareEqual1 = True
				elif ddda1 is copied:
					shouldCompareEqual1 = True

				shouldCompareEqual2 = False
				if ddda2 is original:
					shouldCompareEqual2 = True
				elif ddda2 is same:
					shouldCompareEqual2 = True
				elif ddda2 is copied:
					shouldCompareEqual2 = True

				shouldCompareEqual = shouldCompareEqual1 and shouldCompareEqual2

				if dataSize == 0:
					shouldCompareEqual = True
				elif dataSize == 1:
					tmp1 = ddda1 is missingBeginning or ddda1 is missingEnd
					tmp2 = ddda2 is missingBeginning or ddda2 is missingEnd
					if tmp1 and tmp2:
						shouldCompareEqual = True

				if ddda1 is ddda2:
					shouldCompareEqual = True

				if shouldCompareEqual:
					assert ddda1 == ddda2
					assert ddda1.__eq__(ddda2)
					assert not ddda1 != ddda2
					assert not ddda1.__ne__(ddda2)
				else:
					assert not ddda1 == ddda2
					assert not ddda1.__eq__(ddda2)
					assert ddda1 != ddda2
					assert ddda1.__ne__(ddda2)



		ddda1 = OnTheFlyStatisticsDDDA()
		ddda2 = OnTheFlyStatisticsDDDA()

		ddda1.addDatum(1)
		ddda1.addDatum(2)

		ddda2.addDatum(2)
		ddda2.addDatum(1)

		assert ddda1 == ddda2
		assert ddda1.__eq__(ddda2)
		assert not ddda1 != ddda2
		assert not ddda1.__ne__(ddda2)



def test___repr__():
	dataSizes = \
		[
			0, 1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		]

	for dataSize in dataSizes:
		import random
		data = [random.random() for _ in range(0, dataSize)]

		ddda = OnTheFlyStatisticsDDDA()
		for datum in data:
			ddda.addDatum(datum)

		assert isinstance(ddda.__repr__(), str)
