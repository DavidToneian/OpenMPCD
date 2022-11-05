from MPCDAnalysis.PairPotentials.WeeksChandlerAndersen_DistanceOffset \
	import WeeksChandlerAndersen_DistanceOffset as WCAD

def test_constructor_get_parameters():
	import pytest

	with pytest.raises(TypeError):
		WCAD([1], 1, 1)

	with pytest.raises(TypeError):
		WCAD(1, [1], 1)

	with pytest.raises(TypeError):
		WCAD(1, 1, [1])


	with pytest.raises(ValueError):
		WCAD(-1, 0.1, 2.3)

	with pytest.raises(ValueError):
		WCAD(1, -0.1, 2.3)

	with pytest.raises(ValueError):
		WCAD(1, 0.1, -2.3)


	pot = WCAD(1, 2.5, 3)


	assert isinstance(pot.getEpsilon(), float)
	assert pot.getEpsilon() == 1.0

	assert isinstance(pot.getSigma(), float)
	assert pot.getSigma() == 2.5

	assert isinstance(pot.getD(), float)
	assert pot.getD() == 3.0


def test_getPotential():
	import pytest

	from MPCDAnalysis.Vector3DReal import Vector3DReal

	epsilon = 10.0
	sigma = 2.0
	d = 1.2
	pot = WCAD(epsilon, sigma, d)

	def expected(r):
		if isinstance(r, Vector3DReal):
			r = r.getLength()

		if r <= d:
			return None

		ret = 0.0
		if 2 ** (1.0 / 6.0) * sigma - r + d > 0:
			frac = sigma / (r - d)
			ret = 1.0 / 4.0
			ret += frac ** 12
			ret -= frac ** 6
			ret *= 4 * epsilon

		return ret


	with pytest.raises(TypeError):
		pot.getPotential([1])
	with pytest.raises(ValueError):
		pot.getPotential(-1)

	r = Vector3DReal(0, 0, 0)
	with pytest.raises(ValueError):
		pot.getPotential(r)
	with pytest.raises(ValueError):
		pot.getPotential(r.getLength())

	for i in range(0, 100):
		r = Vector3DReal(-0.1 + i, i * 0.1, 2 * i)

		e = expected(r)
		if e is None:
			with pytest.raises(ValueError):
				pot.getPotential(r)
			continue

		assert pot.getPotential(r) == pytest.approx(e)
		assert pot.getPotential(r.getLength()) == pytest.approx(e)
