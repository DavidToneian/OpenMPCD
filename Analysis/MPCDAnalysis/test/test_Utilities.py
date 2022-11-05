import os
_baseDataPath = os.path.dirname(os.path.abspath(__file__))
_baseDataPath += "/data/test_Utilities/"

def test_openPossiblyCompressedFile():
	from MPCDAnalysis.Utilities import openPossiblyCompressedFile
	import pytest

	import bz2

	def oPCF(suffix):
		prefix = _baseDataPath + "openPossiblyCompressedFile/"
		return openPossiblyCompressedFile(prefix + suffix)

	def compare(suffix, expected):
		assert oPCF(suffix).read() == expected

	def isBZ2File(suffix):
		import bz2
		import sys
		if sys.version_info[0] < 3:
			BZ2FileType = bz2.BZ2File
		else:
			import _io
			BZ2FileType = _io.TextIOWrapper

		assert isinstance(oPCF(suffix), BZ2FileType)

	def isPlainFile(suffix):
		import sys
		if sys.version_info[0] < 3:
			PlainFileType = file
		else:
			import _io
			PlainFileType = _io.TextIOWrapper

		assert isinstance(oPCF(suffix), PlainFileType)



	for x in ["nonexistent.bz2", "dirPath2.bz2"]:
		with pytest.raises(IOError):
			oPCF(x)

	for x in ["nonexistent", "dirPath", "dirPath/foo"]:
		with pytest.raises(ValueError):
			oPCF(x)

	isBZ2File("bz2file.bz2")
	compare("bz2file.bz2", "I am bz2file.bz2!\n")

	isBZ2File("bz2file")
	compare("bz2file.bz2", "I am bz2file.bz2!\n")

	_path = _baseDataPath + "openPossiblyCompressedFile/bz2file"
	assert open(_path, "r").read() == "I am bz2file!\n"

	isPlainFile("plainfile")
	compare("plainfile", "I am plainfile!\n")

	isPlainFile("dirPath/plainfile")
	compare("dirPath/plainfile", "I am dirPath/plainfile!\n")

	isBZ2File("dirPath2.bz2/bz2file.bz2")
	compare("dirPath2.bz2/bz2file.bz2", "I am dirPath2.bz2/bz2file.bz2!\n")

	isBZ2File("dirPath2.bz2/bz2file")
	compare("dirPath2.bz2/bz2file.bz2", "I am dirPath2.bz2/bz2file.bz2!\n")



def test_readValuePairsFromFile():
	from MPCDAnalysis.Utilities import readValuePairsFromFile
	import pytest

	import collections

	def kvp(suffix, minXValue = None):
		prefix = _baseDataPath + "readValuePairsFromFile/"
		if minXValue is None:
			return readValuePairsFromFile(prefix + suffix)
		else:
			return readValuePairsFromFile(prefix + suffix, minXValue)


	for x in ["nonexistent.bz2", "dirPath2.bz2"]:
		with pytest.raises(IOError):
			kvp(x)

	for x in ["nonexistent", "dirPath", "dirPath/foo"]:
		with pytest.raises(ValueError):
			kvp(x)


	with pytest.raises(ValueError):
		kvp("malformed.txt")


	d = kvp("plain.txt")
	assert isinstance(d, collections.OrderedDict)
	expected = collections.OrderedDict()
	expected[-1.5] = 3.1415
	expected[0.0] = 1.0
	expected[900.0] = 0.1
	assert expected == d

	d = kvp("plain.txt", -1.0)
	assert isinstance(d, collections.OrderedDict)
	expected = collections.OrderedDict()
	expected[0.0] = 1.0
	expected[900.0] = 0.1
	assert expected == d

	d = kvp("bz2.txt")
	assert isinstance(d, collections.OrderedDict)
	expected = collections.OrderedDict()
	expected[-1.5] = 3.1415
	expected[0.0] = 1.0
	expected[800.0] = 0.1
	assert expected == d




def test_getConfigValueAndCheckConstistency():
	from MPCDAnalysis.Utilities import getConfigValueAndCheckConstistency as gcc
	import pytest

	def getConfig(suffix):
		prefix = _baseDataPath + "getConfigValueAndCheckConstistency/"

		from MPCDAnalysis.Configuration import Configuration

		return Configuration(prefix + suffix)


	with pytest.raises(TypeError):
		gcc("x", "foo", None)

	with pytest.raises(TypeError):
		gcc("x", "foo", "bar")


	c1 = getConfig("c1.txt")

	with pytest.raises(TypeError):
		gcc(c1, 1, "bar")

	with pytest.raises(TypeError):
		gcc(c1, 1, None)



	assert gcc(c1, "foo", None) == "bar"
	assert gcc(c1, "foo", "bar") == "bar"
	assert gcc(c1, "baz", None) == 1.0
	assert gcc(c1, "baz", 1.0) == 1.0
	assert gcc(c1, "grp.asdf", None) == 10
	assert gcc(c1, "grp.asdf", 10) == 10

	with pytest.raises(ValueError):
		gcc(c1, "foo", "foo")

	with pytest.raises(ValueError):
		gcc(c1, "foo", 1)

	with pytest.raises(KeyError):
		gcc(c1, "asdf", None)

	with pytest.raises(ValueError):
		gcc(c1, "baz", 1234)

	with pytest.raises(ValueError):
		gcc(c1, "baz", "1234")

	with pytest.raises(ValueError):
		gcc(c1, "grp.asdf", -10)



def test_getConsistentConfigValue():
	from MPCDAnalysis.Utilities import getConsistentConfigValue as gcc
	import pytest


	compatible = \
		[
			_baseDataPath + "getConsistentConfigValue/" + x
			for x in ["compatible1", "compatible2"]
		]
	incompatible = \
		[
			_baseDataPath + "getConsistentConfigValue/" + x
			for x in ["compatible1", "compatible2", "other"]
		]


	with pytest.raises(TypeError):
		gcc(compatible, 1)

	with pytest.raises(TypeError):
		gcc((x for x in compatible), "foo")


	assert gcc(compatible, "foo") == "bar"
	assert gcc(compatible, "baz") == 1.0
	assert gcc(compatible, "grp.asdf") == 10

	assert gcc(compatible, "same") == "hello"
	assert gcc(incompatible, "same") == "hello"

	with pytest.raises(ValueError):
		gcc(compatible, "nonexistent")

	with pytest.raises(ValueError):
		gcc(compatible, "unique")

	with pytest.raises(ValueError):
		gcc(incompatible, "foo")

	with pytest.raises(ValueError):
		gcc(incompatible, "baz")

	with pytest.raises(ValueError):
		gcc(incompatible, "grp.asdf")


def test_getNumberOfArgumentsFromCallable():
	from MPCDAnalysis.Utilities import getNumberOfArgumentsFromCallable as gna
	import pytest

	def zeroArgs():
		pass

	def oneArg(arg1):
		pass

	def twoArgsWithDefault(arg1, arg2 = 1):
		pass

	def threeArgs(arg1, arg2, arg3):
		pass

	fourArgs = lambda x, y, z, w: 0


	assert gna(zeroArgs) == 0
	assert gna(oneArg) == 1
	assert gna(twoArgsWithDefault) == 2
	assert gna(threeArgs) == 3
	assert gna(fourArgs) == 4
	assert gna(lambda a, b, c, d, e: a) == 5
