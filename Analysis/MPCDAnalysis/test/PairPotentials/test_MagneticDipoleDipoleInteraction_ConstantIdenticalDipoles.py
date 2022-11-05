from MPCDAnalysis.PairPotentials.MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles \
	import MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles as Potential

def test_constructor_get_parameters():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import pytest

	orientation = Vector3DReal(1, 0, 0)

	with pytest.raises(TypeError):
		Potential([1], orientation)

	with pytest.raises(TypeError):
		Potential(1, [1, 0, 0])


	with pytest.raises(ValueError):
		Potential(-1, orientation)

	with pytest.raises(ValueError):
		Potential(1, orientation * 2)


	pot = Potential(2.5, orientation)


	assert isinstance(pot.getPrefactor(), float)
	assert pot.getPrefactor() == 2.5

	assert pot.getOrientation() == orientation

	orientation.x = 2
	assert pot.getOrientation() == Vector3DReal(1, 0, 0)


def test_getPotential():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import pytest

	prefactor = 2.5
	orientation = Vector3DReal(0, 1, 0)
	pot = Potential(prefactor, orientation)

	def expected(r):
		ret = -prefactor
		ret *= 3 * orientation.dot(r.getNormalized()) ** 2 - 1
		ret /= r.getLength() ** 3

		return ret


	with pytest.raises(TypeError):
		pot.getPotential([1])
	with pytest.raises(TypeError):
		pot.getPotential(1)

	r = Vector3DReal(0, 0, 0)
	with pytest.raises(ValueError):
		pot.getPotential(r)

	for i in range(0, 100):
		r = Vector3DReal(-0.1 + i, i * 0.1, 2 * i)

		assert pot.getPotential(r) == pytest.approx(expected(r))
