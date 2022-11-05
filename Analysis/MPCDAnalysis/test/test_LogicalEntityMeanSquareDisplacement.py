import pytest

from MPCDAnalysis.LogicalEntityMeanSquareDisplacement \
	import LogicalEntityMeanSquareDisplacement

def test___init__():
	import os
	dataPath = os.path.dirname(os.path.abspath(__file__))
	dataPath += "/data/"

	with pytest.raises(TypeError):
		LogicalEntityMeanSquareDisplacement(0)

	with pytest.raises(ValueError):
		LogicalEntityMeanSquareDisplacement(dataPath + "nonexistent")

	with pytest.raises(ValueError):
		LogicalEntityMeanSquareDisplacement([])


def test_referenceData():
	import os
	dataPath = os.path.dirname(os.path.abspath(__file__))
	dataPath += "/data/test_LogicalEntityMeanSquareDisplacement/"

	runPath1 = dataPath + "run1"
	runPath2 = dataPath + "run2"

	expectedPath1 = dataPath + "expected1"
	expectedPath2 = dataPath + "expected2"
	expectedPath12 = dataPath + "expected12"


	msd1 = LogicalEntityMeanSquareDisplacement(runPath1)
	msd2 = LogicalEntityMeanSquareDisplacement(runPath2)
	msd12 = LogicalEntityMeanSquareDisplacement([runPath1, runPath2])

	assert msd1.getMaximumMeasurementTime() == 1000
	assert msd2.getMaximumMeasurementTime() == 1000
	assert msd12.getMaximumMeasurementTime() == 1000

	results1 = {}
	for deltaT in range(1, msd1.getMaximumMeasurementTime() + 1):
		results1[deltaT] = msd1.getMeanSquareDisplacement(deltaT)
	results2 = {}
	for deltaT in range(1, msd2.getMaximumMeasurementTime() + 1):
		results2[deltaT] = msd2.getMeanSquareDisplacement(deltaT)
	results12 = {}
	for deltaT in range(1, msd12.getMaximumMeasurementTime() + 1):
		results12[deltaT] = msd12.getMeanSquareDisplacement(deltaT)


	import pickle

	#to create the files containing expeted values:
	#with open(expectedPath1, "wb") as f:
	#	pickle.dump(results1, f)
	#with open(expectedPath2, "wb") as f:
	#	pickle.dump(results2, f)
	#with open(expectedPath12, "wb") as f:
	#	pickle.dump(results12, f)

	with open(expectedPath1, "rb") as f:
		expected1 = pickle.load(f)
	with open(expectedPath2, "rb") as f:
		expected2 = pickle.load(f)
	with open(expectedPath12, "rb") as f:
		expected12 = pickle.load(f)


	assert expected1 == results1
	assert expected2 == results2
	assert expected12 == results12
