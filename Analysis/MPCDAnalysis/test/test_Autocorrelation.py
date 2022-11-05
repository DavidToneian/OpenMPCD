import pytest

from MPCDAnalysis.Autocorrelation import Autocorrelation
from MPCDAnalysis.OnTheFlyStatisticsDDDA import OnTheFlyStatisticsDDDA

def test_constructor():
	with pytest.raises(TypeError):
		Autocorrelation(1.0)

	with pytest.raises(ValueError):
		Autocorrelation(-1)

	Autocorrelation(0)
	Autocorrelation(1)
	Autocorrelation(2)


def test_getMaxCorrelationTime():
	for N_max in range(0, 5):
		ac = Autocorrelation(N_max)
		assert ac.getMaxCorrelationTime() == N_max


def test_correlationTimeIsAvailable():
	N_max = 10

	def testExceptions(ac):
		for N in range(0, ac.getMaxCorrelationTime() + 1):
			with pytest.raises(TypeError):
				ac.correlationTimeIsAvailable(float(N))

		with pytest.raises(ValueError):
			ac.correlationTimeIsAvailable(-1)

		with pytest.raises(ValueError):
			ac.correlationTimeIsAvailable(ac.getMaxCorrelationTime() + 1)



	ac = Autocorrelation(N_max)

	testExceptions(ac)

	for datum in range(0, N_max * 2):
		dataPointsSuppliedSoFar = datum
		for N in range(0, N_max + 1):
			if dataPointsSuppliedSoFar >= N + 1:
				expected = True
			else:
				expected = False
			assert ac.correlationTimeIsAvailable(N) == expected

		ac.addDatum(datum)

	testExceptions(ac)


def test_getAutocorrelation():
	import random
	n_max = 100
	N_max = 10

	data = [random.uniform(-10, 10) for _ in range(0, n_max)]

	ac = Autocorrelation(N_max)

	for datumIndex, datum in enumerate(data):
		ac.addDatum(datum)

		dataSupplied = data[0 : datumIndex + 1]
		for N in range(0, N_max + 1):
			if not ac.correlationTimeIsAvailable(N):
				continue

			ddda = OnTheFlyStatisticsDDDA()
			for i in range(0, len(dataSupplied)):
				if i + N >= len(dataSupplied):
					break
				ddda.addDatum(dataSupplied[i] * dataSupplied[i + N])

			assert ac.getAutocorrelation(N) == ddda


def test_getAutocorrelation_Vector3DReal():
	import random
	n_max = 100
	N_max = 10

	from MPCDAnalysis.Vector3DReal import Vector3DReal
	data = [
		Vector3DReal(
			random.uniform(-10, 10),
			random.uniform(-10, 10),
			random.uniform(-10, 10))
		for _ in range(0, n_max)]

	ac = Autocorrelation(N_max)

	def mul(v1, v2):
		return v1.dot(v2)

	for datumIndex, datum in enumerate(data):
		ac.addDatum(datum, mul)

		dataSupplied = data[0 : datumIndex + 1]
		for N in range(0, N_max + 1):
			if not ac.correlationTimeIsAvailable(N):
				continue

			ddda = OnTheFlyStatisticsDDDA()
			for i in range(0, len(dataSupplied)):
				if i + N >= len(dataSupplied):
					break
				ddda.addDatum(dataSupplied[i].dot(dataSupplied[i + N]))

			assert ac.getAutocorrelation(N) == ddda
