from MPCDAnalysis.PairPotentials.FENE import FENE

def test_constructor_getK_get_l_0_getR():
	import pytest

	with pytest.raises(TypeError):
		FENE([1], 1, 1)

	with pytest.raises(TypeError):
		FENE(1, [1], 1)

	with pytest.raises(TypeError):
		FENE(1, 1, [1])

	with pytest.raises(ValueError):
		FENE(1, 1, 0)

	with pytest.raises(ValueError):
		FENE(1, 1, -1.5)

	fene = FENE(1, 2.5, 3)


	assert isinstance(fene.getK(), float)
	assert fene.getK() == 1.0

	assert isinstance(fene.get_l_0(), float)
	assert fene.get_l_0() == 2.5

	assert isinstance(fene.getR(), float)
	assert fene.getR() == 3.0


def test_getPotential():
	import math
	import pytest

	from MPCDAnalysis.Vector3DReal import Vector3DReal

	K = 2.0
	R = 10.0
	l_0 = 3.0
	fene = FENE(K, l_0, R)

	with pytest.raises(TypeError):
		fene.getPotential([1])
	with pytest.raises(ValueError):
		fene.getPotential(-1)

	with pytest.raises(ValueError):
		fene.getPotential(Vector3DReal(l_0 + R, 0, 0))
	with pytest.raises(ValueError):
		fene.getPotential(l_0 + R)
	with pytest.raises(ValueError):
		fene.getPotential(Vector3DReal(l_0 + R + 0.1, 0, 0))
	with pytest.raises(ValueError):
		fene.getPotential(l_0 + R + 0.1)

	r = Vector3DReal(0, 0, 0)
	expected = -0.5 * K * R * R * math.log(1 - (l_0 / R) ** 2)
	assert fene.getPotential(r) == expected
	assert fene.getPotential(r.getLength()) == expected

	for i in range(0, 100):
		r = Vector3DReal(-0.1 + i, i * 0.1, 2 * i)
		frac = (r.getLength() - l_0) / R

		if frac ** 2 >= 1:
			with pytest.raises(ValueError):
				fene.getPotential(r)
			continue

		expected = -0.5 * K * R * R * math.log(1 - frac ** 2);
		assert fene.getPotential(r) == expected
		assert fene.getPotential(r.getLength()) == expected
