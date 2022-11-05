from __future__ import division

import math
import pytest

from MPCDAnalysis.OnTheFlyStatistics import OnTheFlyStatistics

def test_emptyInstance():
	stat = OnTheFlyStatistics()

	assert stat.getSampleSize() == 0

	with pytest.raises(Exception):
		stat.getSampleMean()

	with pytest.raises(Exception):
		stat.getSampleVariance()


def test_oneDatum():
	stat = OnTheFlyStatistics()

	stat.addDatum(-1)

	assert stat.getSampleSize() == 1
	assert stat.getSampleMean() == -1

	with pytest.raises(Exception):
		stat.getSampleVariance()


def test_twoData_int():
	stat = OnTheFlyStatistics()

	data = [-1, 0]

	for datum in data:
		stat.addDatum(datum)

	assert stat.getSampleSize() == 2
	assert stat.getSampleMean() == -0.5
	assert stat.getSampleVariance() == 0.5
	assert \
		math.sqrt(stat.getSampleVariance()) == stat.getSampleStandardDeviation()
	assert \
		stat.getStandardErrorOfTheMean() == \
		stat.getSampleStandardDeviation() / math.sqrt(stat.getSampleSize())


def test_twoData_float():
	stat = OnTheFlyStatistics()

	data = [5, 2.0]

	for datum in data:
		stat.addDatum(datum)

	assert stat.getSampleSize() == 2
	assert stat.getSampleMean() == sum(data) / len(data)
	assert stat.getSampleVariance() == 4.5
	assert \
		math.sqrt(stat.getSampleVariance()) == stat.getSampleStandardDeviation()
	assert \
		stat.getStandardErrorOfTheMean() == \
		stat.getSampleStandardDeviation() / math.sqrt(stat.getSampleSize())


def test_threeData():
	stat = OnTheFlyStatistics()

	data = [-1, 0, 1.0]

	for datum in data:
		stat.addDatum(datum)

	assert stat.getSampleSize() == 3
	assert stat.getSampleMean() == 0
	assert stat.getSampleVariance() == 1
	assert \
		math.sqrt(stat.getSampleVariance()) == stat.getSampleStandardDeviation()
	assert \
		stat.getStandardErrorOfTheMean() == \
		stat.getSampleStandardDeviation() / math.sqrt(stat.getSampleSize())


def test_mergeSample():
	stat = OnTheFlyStatistics()

	data = [-1, 0, 1.0]

	for datum in data:
		stat.addDatum(datum)

	import copy
	stat1 = copy.deepcopy(stat)
	stat2 = OnTheFlyStatistics()

	stat.addDatum(3)
	stat2.addDatum(3)

	stat.addDatum(4.5)
	stat2.addDatum(4.5)

	stat1.mergeSample(stat2)

	assert stat1.getSampleSize() == stat.getSampleSize()
	assert stat1.getSampleMean() == stat.getSampleMean()
	assert stat1.getSampleVariance() == stat.getSampleVariance()
	assert \
		stat1.getSampleStandardDeviation() == stat.getSampleStandardDeviation()
	assert \
		stat1.getStandardErrorOfTheMean() == stat.getStandardErrorOfTheMean()


def test_mergeSamples():
	stat = OnTheFlyStatistics()

	data = [-1, 0, 1.0]

	for datum in data:
		stat.addDatum(datum)

	import copy
	stat1 = copy.deepcopy(stat)
	stat2 = OnTheFlyStatistics()

	stat.addDatum(3)
	stat2.addDatum(3)

	stat.addDatum(4.5)
	stat2.addDatum(4.5)

	stat1.mergeSample(stat2)

	stat_threeMerged = copy.deepcopy(stat)
	stat_threeMerged.mergeSamples([stat1, stat2])

	stat.mergeSample(stat1)
	stat.mergeSample(stat2)

	assert stat_threeMerged.getSampleSize() == stat.getSampleSize()
	assert stat_threeMerged.getSampleMean() == stat.getSampleMean()
	assert stat_threeMerged.getSampleVariance() == stat.getSampleVariance()
	assert \
		stat_threeMerged.getSampleStandardDeviation() == \
		stat.getSampleStandardDeviation()
	assert \
		stat_threeMerged.getStandardErrorOfTheMean() == \
		stat.getStandardErrorOfTheMean()


def test_constructor_with_arguments():
	stat = OnTheFlyStatistics(3, 4, 5)
	assert stat.getSampleSize() == 5
	assert stat.getSampleMean() == 3
	assert stat.getSampleVariance() == 4
	assert \
		stat.getSampleStandardDeviation() == math.sqrt(stat.getSampleVariance())
	assert \
		stat.getStandardErrorOfTheMean() == \
		stat.getSampleStandardDeviation() / math.sqrt(stat.getSampleSize())

	stat2 = OnTheFlyStatistics()
	stat2.mergeSample(stat)

	assert stat2.getSampleSize() == stat.getSampleSize()
	assert stat2.getSampleMean() == stat.getSampleMean()
	assert stat2.getSampleVariance() == stat.getSampleVariance()
	assert \
		stat2.getStandardErrorOfTheMean() == \
		stat.getStandardErrorOfTheMean()


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

		stat = OnTheFlyStatistics()
		for datum in data:
			stat.addDatum(datum)
			assert isinstance(stat.serializeToString(), str)


def test_unserializeFromString():
	stat = OnTheFlyStatistics()

	with pytest.raises(TypeError):
		stat.unserializeFromString(["foo"])
	with pytest.raises(TypeError):
		stat.unserializeFromString(stat)

	with pytest.raises(ValueError):
		stat.unserializeFromString("")
	with pytest.raises(ValueError):
		stat.unserializeFromString("foo")
	with pytest.raises(ValueError):
		stat.unserializeFromString("123;0;0;0")
	with pytest.raises(ValueError):
		stat.unserializeFromString("1;0;1;0")
	with pytest.raises(ValueError):
		stat.unserializeFromString("1;0;0;1")
	with pytest.raises(ValueError):
		stat.unserializeFromString("1;-1;0.0;0.0")
	with pytest.raises(ValueError):
		stat.unserializeFromString("1;2;0;-1")
	with pytest.raises(ValueError):
		stat.unserializeFromString("1;0;0;0;0")
	with pytest.raises(ValueError):
		stat.unserializeFromString("1;0;0;0;")
	with pytest.raises(ValueError):
		stat.unserializeFromString("1;0;;0")


	stat.unserializeFromString("1;0;0;0")
	assert stat.getSampleSize() == 0

	stat = OnTheFlyStatistics()
	unserialized = OnTheFlyStatistics()
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

		if lhs.getSampleSize() == 1:
			return True

		if lhs.getSampleVariance() != pytest.approx(rhs.getSampleVariance()):
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


	stat = OnTheFlyStatistics()
	stat.addDatum(1)
	stat.addDatum(2)
	stat.addDatum(3)
	stat.addDatum(4)
	stat.addDatum(5)

	unserialized = OnTheFlyStatistics()
	unserialized.unserializeFromString("1;5;3.0;10.0")
	assert approximatelyEqual(unserialized, stat)



def test___eq_____ne_____hash__():
	stat1_1 = OnTheFlyStatistics(3, 4, 5)
	stat1_2 = OnTheFlyStatistics(3, 4, 5)
	stat2 = OnTheFlyStatistics(1, 4, 5)
	stat3 = OnTheFlyStatistics(3, 1, 5)
	stat4 = OnTheFlyStatistics(3, 4, 1)

	stat2to4 = [stat2, stat3, stat4]
	allstat = [stat1_1, stat1_2] + stat2to4

	for stat in allstat:
		assert stat is stat
		assert stat == stat
		assert not stat != stat
		assert stat.__hash__() == stat.__hash__()

	assert stat1_1 == stat1_2
	assert stat1_2 == stat1_1
	assert not stat1_1 != stat1_2
	assert not stat1_2 != stat1_1
	assert stat1_1.__hash__() == stat1_2.__hash__()

	for statcollection in [[stat1_1] + stat2to4, [stat1_2] + stat2to4]:
		for stat in statcollection:
			for other in statcollection:
				if stat is other:
					continue
				assert not stat == other
				assert stat != other
				assert stat.__hash__() != other.__hash__()


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

		stat = OnTheFlyStatistics()
		for datum in data:
			stat.addDatum(datum)

		assert isinstance(stat.__repr__(), str)
