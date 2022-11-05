from MPCDAnalysis.ParticleCollection import ParticleCollection

def test___init__():
	ParticleCollection()


def test_setPositionsAndVelocities_getPositions_getVelocities():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import pytest


	nullVector = Vector3DReal(0.0, 0.0, 0.0)

	pc = ParticleCollection()
	pc.setPositionsAndVelocities([nullVector], [nullVector])

	with pytest.raises(TypeError):
		pc = ParticleCollection()
		pc.setPositionsAndVelocities((nullVector,), [nullVector])
	with pytest.raises(TypeError):
		pc = ParticleCollection()
		pc.setPositionsAndVelocities([nullVector], (nullVector,))

	with pytest.raises(TypeError):
		pc = ParticleCollection()
		pc.setPositionsAndVelocities([[0.0, 0.0, 0.0]], [nullVector])
	with pytest.raises(TypeError):
		pc = ParticleCollection()
		pc.setPositionsAndVelocities([nullVector], [[0.0, 0.0, 0.0]])


	with pytest.raises(ValueError):
		pc = ParticleCollection()
		pc.setPositionsAndVelocities([nullVector, nullVector], [nullVector])
	with pytest.raises(ValueError):
		pc = ParticleCollection()
		pc.setPositionsAndVelocities([nullVector], [nullVector, nullVector])


	pc = ParticleCollection()
	positions = []
	velocities = []
	for x in range(0, 5):
		positions.append(Vector3DReal(0.5 * x, -0.5 + x, x * x / 2.0))
		velocities.append(Vector3DReal(1.23 / (x + 1), x, -x))

	pc.setPositionsAndVelocities(positions, velocities)

	assert isinstance(pc.getPositions(), list)
	assert isinstance(pc.getVelocities(), list)
	for x in range(0, 5):
		assert isinstance(pc.getPositions()[x], Vector3DReal)
		assert isinstance(pc.getVelocities()[x], Vector3DReal)

		assert \
			pc.getPositions()[x] == Vector3DReal(0.5 * x, -0.5 + x, x * x / 2.0)
		assert \
			pc.getVelocities()[x] == Vector3DReal(1.23 / (x + 1), x, -x)


	positions[0].normalize()
	velocities[1].normalize()
	assert isinstance(pc.getPositions(), list)
	assert isinstance(pc.getVelocities(), list)
	for x in range(0, 5):
		assert isinstance(pc.getPositions()[x], Vector3DReal)
		assert isinstance(pc.getVelocities()[x], Vector3DReal)

		assert \
			pc.getPositions()[x] == Vector3DReal(0.5 * x, -0.5 + x, x * x / 2.0)
		assert \
			pc.getVelocities()[x] == Vector3DReal(1.23 / (x + 1), x, -x)


def test_setUniformMass():
	import pytest

	pc = ParticleCollection()
	with pytest.raises(TypeError):
		pc.setUniformMass([1.0])

	pc = ParticleCollection()
	with pytest.raises(ValueError):
		pc.setUniformMass(-0.5)

	pc = ParticleCollection()
	pc.setUniformMass(0.0)
	pc.setUniformMass(1.0)
	pc.setUniformMass(1.5)
	pc.setUniformMass(0)
	pc.setUniformMass(1)

	from MPCDAnalysis.Vector3DReal import Vector3DReal
	nullVector = Vector3DReal(0, 0, 0)

	pc.setPositionsAndVelocities([nullVector], [nullVector])
	pc.setUniformMass(0.0)
	pc.setUniformMass(1.0)
	pc.setUniformMass(1.5)
	pc.setUniformMass(0)
	pc.setUniformMass(1)


def test_isEmpty():
	pc = ParticleCollection()
	assert pc.isEmpty()
	assert pc.isEmpty()

	from MPCDAnalysis.Vector3DReal import Vector3DReal
	nullVector = Vector3DReal(0, 0, 0)

	pc.setPositionsAndVelocities([nullVector], [nullVector])

	assert not pc.isEmpty()
	assert not pc.isEmpty()

	twice = [nullVector, nullVector]
	pc.setPositionsAndVelocities(twice, twice)

	assert not pc.isEmpty()
	assert not pc.isEmpty()


def test_getParticleCount():
	pc = ParticleCollection()
	assert pc.getParticleCount() == 0
	assert pc.getParticleCount() == 0
	assert isinstance(pc.getParticleCount(), int)

	from MPCDAnalysis.Vector3DReal import Vector3DReal
	nullVector = Vector3DReal(0, 0, 0)

	pc.setPositionsAndVelocities([nullVector], [nullVector])

	assert pc.getParticleCount() == 1
	assert pc.getParticleCount() == 1
	assert isinstance(pc.getParticleCount(), int)

	twice = [nullVector, nullVector]
	pc.setPositionsAndVelocities(twice, twice)

	assert pc.getParticleCount() == 2
	assert pc.getParticleCount() == 2
	assert isinstance(pc.getParticleCount(), int)


def test_getPosition_getVelocity():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import pytest


	pc = ParticleCollection()
	with pytest.raises(IndexError):
		pc.getPosition(0)
	with pytest.raises(IndexError):
		pc.getPosition(-1)
	with pytest.raises(IndexError):
		pc.getVelocity(0)
	with pytest.raises(IndexError):
		pc.getVelocity(-1)


	pc = ParticleCollection()
	positions = []
	velocities = []
	for x in range(0, 5):
		positions.append(Vector3DReal(0.5 * x, -0.5 + x, x * x / 2.0))
		velocities.append(Vector3DReal(1.23 / (x + 1), x, -x))

	pc.setPositionsAndVelocities(positions, velocities)


	with pytest.raises(TypeError):
		pc.getPosition(0.0)
	with pytest.raises(TypeError):
		pc.getVelocity(0.0)

	with pytest.raises(IndexError):
		pc.getPosition(-1)
	with pytest.raises(IndexError):
		pc.getPosition(pc.getParticleCount())
	with pytest.raises(IndexError):
		pc.getPosition(pc.getParticleCount() + 1)

	with pytest.raises(IndexError):
		pc.getVelocity(-1)
	with pytest.raises(IndexError):
		pc.getVelocity(pc.getParticleCount())
	with pytest.raises(IndexError):
		pc.getVelocity(pc.getParticleCount() + 1)


	for x in range(0, 5):
		assert isinstance(pc.getPosition(x), Vector3DReal)
		assert isinstance(pc.getVelocity(x), Vector3DReal)

		assert \
			pc.getPosition(x) == Vector3DReal(0.5 * x, -0.5 + x, x * x / 2.0)
		assert \
			pc.getVelocity(x) == Vector3DReal(1.23 / (x + 1), x, -x)


	positions[0].normalize()
	velocities[1].normalize()
	for x in range(0, 5):
		assert isinstance(pc.getPosition(x), Vector3DReal)
		assert isinstance(pc.getVelocity(x), Vector3DReal)

		assert \
			pc.getPosition(x) == Vector3DReal(0.5 * x, -0.5 + x, x * x / 2.0)
		assert \
			pc.getVelocity(x) == Vector3DReal(1.23 / (x + 1), x, -x)


def test_getMass():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import pytest


	pc = ParticleCollection()
	with pytest.raises(IndexError):
		pc.getMass(0)
	with pytest.raises(IndexError):
		pc.getMass(-1)


	pc = ParticleCollection()
	positions = []
	velocities = []
	for x in range(0, 5):
		positions.append(Vector3DReal(0.5 * x, -0.5 + x, x * x / 2.0))
		velocities.append(Vector3DReal(1.23 / (x + 1), x, -x))

	pc.setPositionsAndVelocities(positions, velocities)

	with pytest.raises(RuntimeError):
		pc.getMass(0)


	with pytest.raises(TypeError):
		pc.getMass(0.0)
	with pytest.raises(TypeError):
		pc.getMass(0.0)

	with pytest.raises(IndexError):
		pc.getMass(-1)
	with pytest.raises(IndexError):
		pc.getMass(pc.getParticleCount())
	with pytest.raises(IndexError):
		pc.getMass(pc.getParticleCount() + 1)


	pc.setUniformMass(1.23)

	for x in range(0, pc.getParticleCount()):
		assert isinstance(pc.getMass(x), float)
		assert pc.getMass(x) == 1.23

	pc.setPositionsAndVelocities(positions[:3], velocities[:3])
	for x in range(0, pc.getParticleCount()):
		assert isinstance(pc.getMass(x), float)
		assert pc.getMass(x) == 1.23

	pc.setPositionsAndVelocities(positions + velocities, velocities + positions)
	for x in range(0, pc.getParticleCount()):
		assert isinstance(pc.getMass(x), float)
		assert pc.getMass(x) == 1.23

	pc.setUniformMass(4)
	for x in range(0, pc.getParticleCount()):
		assert isinstance(pc.getMass(x), float)
		assert pc.getMass(x) == 4


def test_getMomentum():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import pytest


	pc = ParticleCollection()
	with pytest.raises(IndexError):
		pc.getMomentum(0)
	with pytest.raises(IndexError):
		pc.getMomentum(-1)


	pc = ParticleCollection()
	positions = []
	velocities = []
	for x in range(0, 5):
		positions.append(Vector3DReal(0.5 * x, -0.5 + x, x * x / 2.0))
		velocities.append(Vector3DReal(1.23 / (x + 1), x, -x))

	pc.setPositionsAndVelocities(positions, velocities)

	with pytest.raises(RuntimeError):
		pc.getMomentum(0)


	with pytest.raises(TypeError):
		pc.getMomentum(0.0)
	with pytest.raises(TypeError):
		pc.getMomentum(0.0)

	with pytest.raises(IndexError):
		pc.getMomentum(-1)
	with pytest.raises(IndexError):
		pc.getMomentum(pc.getParticleCount())
	with pytest.raises(IndexError):
		pc.getMomentum(pc.getParticleCount() + 1)


	pc.setUniformMass(1.23)

	for x in range(0, pc.getParticleCount()):
		assert isinstance(pc.getMomentum(x), Vector3DReal)
		assert pc.getMomentum(x) == pc.getVelocity(x) * pc.getMass(x)
		assert pc.getMomentum(x) == pc.getVelocity(x) * pc.getMass(x)

	pc.setPositionsAndVelocities(positions[:3], velocities[:3])
	for x in range(0, pc.getParticleCount()):
		assert isinstance(pc.getMomentum(x), Vector3DReal)
		assert pc.getMomentum(x) == pc.getVelocity(x) * pc.getMass(x)
		assert pc.getMomentum(x) == pc.getVelocity(x) * pc.getMass(x)

	pc.setPositionsAndVelocities(positions + velocities, velocities + positions)
	for x in range(0, pc.getParticleCount()):
		assert isinstance(pc.getMomentum(x), Vector3DReal)
		assert pc.getMomentum(x) == pc.getVelocity(x) * pc.getMass(x)
		assert pc.getMomentum(x) == pc.getVelocity(x) * pc.getMass(x)

	pc.setUniformMass(4.56)
	for x in range(0, pc.getParticleCount()):
		assert isinstance(pc.getMomentum(x), Vector3DReal)
		assert pc.getMomentum(x) == pc.getVelocity(x) * pc.getMass(x)
		assert pc.getMomentum(x) == pc.getVelocity(x) * pc.getMass(x)


def test_getCenterOfMass():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import pytest


	pc = ParticleCollection()
	with pytest.raises(RuntimeError):
		pc.getCenterOfMass()

	positions = []
	velocities = []
	for x in range(0, 5):
		positions.append(Vector3DReal(0.5 * x, -0.5 + x, x * x / 2.0))
		velocities.append(Vector3DReal(1.23 / (x + 1), x, -x))

	pc.setPositionsAndVelocities(positions, velocities)

	with pytest.raises(RuntimeError):
		pc.getCenterOfMass()

	mass = 1.23
	pc.setUniformMass(mass)

	expected = Vector3DReal(0.0, 0.0, 0.0)
	totalMass = 0.0
	for position in positions:
		expected += position * mass
		totalMass += mass
	expected /= totalMass

	assert isinstance(pc.getCenterOfMass(), Vector3DReal)
	assert pc.getCenterOfMass() == expected
	assert pc.getCenterOfMass() == expected


def test_getCenterOfPositions():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import pytest


	pc = ParticleCollection()
	with pytest.raises(RuntimeError):
		pc.getCenterOfPositions()

	positions = []
	velocities = []
	for x in range(0, 5):
		positions.append(Vector3DReal(0.5 * x, -0.5 + x, x * x / 2.0))
		velocities.append(Vector3DReal(1.23 / (x + 1), x, -x))

	pc.setPositionsAndVelocities(positions, velocities)

	expected = Vector3DReal(0.0, 0.0, 0.0)
	for position in positions:
		expected += position
	expected /= len(positions)

	assert isinstance(pc.getCenterOfPositions(), Vector3DReal)
	assert pc.getCenterOfPositions() == expected
	assert pc.getCenterOfPositions() == expected


def test_getCenterOfMassVelocity():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import pytest


	pc = ParticleCollection()
	with pytest.raises(RuntimeError):
		pc.getCenterOfMassVelocity()

	positions = []
	velocities = []
	for x in range(0, 5):
		positions.append(Vector3DReal(0.5 * x, -0.5 + x, x * x / 2.0))
		velocities.append(Vector3DReal(1.23 / (x + 1), x, -x))

	pc.setPositionsAndVelocities(positions, velocities)

	with pytest.raises(RuntimeError):
		pc.getCenterOfMassVelocity()

	mass = 1.23
	pc.setUniformMass(mass)

	expected = Vector3DReal(0.0, 0.0, 0.0)
	totalMass = 0.0
	for velocity in velocities:
		expected += velocity * mass
		totalMass += mass
	expected /= totalMass

	assert isinstance(pc.getCenterOfMassVelocity(), Vector3DReal)
	assert pc.getCenterOfMassVelocity() == expected
	assert pc.getCenterOfMassVelocity() == expected


def test_rotateAroundNormalizedAxis():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import pytest

	axis = Vector3DReal(1, 2, 3)
	axis.normalize()
	angle = -2.34

	pc = ParticleCollection()
	pc.rotateAroundNormalizedAxis(axis, angle)

	with pytest.raises(TypeError):
		pc.rotateAroundNormalizedAxis([1.0, 0.0, 0.0], angle)
	with pytest.raises(TypeError):
		pc.rotateAroundNormalizedAxis(axis, 1)
	with pytest.raises(ValueError):
		pc.rotateAroundNormalizedAxis(Vector3DReal(0, 0, 0), angle)

	positions = []
	velocities = []
	for x in range(0, 5):
		positions.append(Vector3DReal(0.5 * x, -0.5 + x, x * x / 2.0))
		velocities.append(Vector3DReal(1.23 / (x + 1), x, -x))

	pc.setPositionsAndVelocities(positions, velocities)

	pc.rotateAroundNormalizedAxis(axis, angle)

	for x in range(0, pc.getParticleCount()):
		expectedPos = positions[x].getRotatedAroundNormalizedAxis(axis, angle)
		expectedVel = velocities[x].getRotatedAroundNormalizedAxis(axis, angle)

		assert pc.getPosition(x) == expectedPos
		assert pc.getPosition(x) == expectedPos

		assert pc.getVelocity(x) == expectedVel
		assert pc.getVelocity(x) == expectedVel


def test_getGyrationTensor():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import numpy
	import pytest


	pc = ParticleCollection()
	with pytest.raises(ValueError):
		pc.getGyrationTensor()
	with pytest.raises(ValueError):
		pc.getGyrationTensor()

	particleCount = 5
	positions = []
	velocities = []
	for x in range(0, particleCount):
		positions.append(Vector3DReal(0.5 * x, -0.5 + x, x * x / 2.0))
		velocities.append(Vector3DReal(1.23 / (x + 1), x, -x))

	pc.setPositionsAndVelocities(positions, velocities)

	expected = numpy.zeros((3, 3))
	for m in range(0, 3):
		for n in range(0, 3):
			for i in range(0, len(positions)):
				for j in range(0, len(positions)):
					r1 = positions[i]
					r2 = positions[j]
					expected[m][n] += (r1[m] - r2[m]) * (r1[n] - r2[n])

	for i in range(0, 3):
		for j in range(0, 3):
			expected[i][j] /= (2 * particleCount * particleCount)

	S = pc.getGyrationTensor()
	S2 = pc.getGyrationTensor()

	assert numpy.array_equal(S, S2)

	assert isinstance(S, numpy.ndarray)
	assert S.shape == (3, 3)
	for i in range(0, 3):
		for j in range(0, 3):
			assert S[i][j] == S[j][i]

	for i in range(0, 3):
		for j in range(0, 3):
			assert S[i][j] == pytest.approx(expected[i][j])


def test_getGyrationTensorPrincipalMoments():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import numpy
	import pytest


	pc = ParticleCollection()
	with pytest.raises(ValueError):
		pc.getGyrationTensor()
	with pytest.raises(ValueError):
		pc.getGyrationTensor()

	particleCount = 5
	positions = []
	velocities = []
	for x in range(0, particleCount):
		positions.append(Vector3DReal(0.5 * x, -0.5 + x, x * x / 2.0))
		velocities.append(Vector3DReal(1.23 / (x + 1), x, -x))

	pc.setPositionsAndVelocities(positions, velocities)

	eigensystem = pc.getGyrationTensorEigensystem()

	eigenvalues1 = pc.getGyrationTensorPrincipalMoments()
	eigenvalues2 = pc.getGyrationTensorPrincipalMoments()

	for eigenvalues in [eigenvalues1, eigenvalues2]:
		assert isinstance(eigenvalues, list)
		assert len(eigenvalues) == 3

		for i in range(0, 3):
			assert isinstance(eigenvalues[i], numpy.float64)
			assert numpy.isreal(eigenvalues[i])

			assert eigenvalues[i] == eigensystem[i][0]


def test_getGyrationTensorEigensystem():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import numpy
	import pytest


	pc = ParticleCollection()
	with pytest.raises(ValueError):
		pc.getGyrationTensor()
	with pytest.raises(ValueError):
		pc.getGyrationTensor()

	particleCount = 5
	positions = []
	velocities = []
	for x in range(0, particleCount):
		positions.append(Vector3DReal(0.5 * x, -0.5 + x, x * x / 2.0))
		velocities.append(Vector3DReal(1.23 / (x + 1), x, -x))

	pc.setPositionsAndVelocities(positions, velocities)

	eigensystem1 = pc.getGyrationTensorEigensystem()
	eigensystem2 = pc.getGyrationTensorEigensystem()

	gyrationTensor = pc.getGyrationTensor()

	for eigensystem in [eigensystem1, eigensystem2]:
		assert isinstance(eigensystem, list)
		assert len(eigensystem) == 3

		previous = None
		for eigenpair in eigensystem:
			assert isinstance(eigenpair, list)
			assert len(eigenpair) == 2

			eigenvalue = eigenpair[0]
			eigenvector = eigenpair[1]

			assert numpy.isreal(eigenvalue)
			assert isinstance(eigenvalue, numpy.float64)

			if previous is None:
				previous = eigenvalue
			else:
				assert previous < eigenvalue
				previous = eigenvalue

			assert eigenvector.shape == (3,)
			for i in range(0, 3):
				assert isinstance(eigenvector[i], numpy.float64)


			Sv = gyrationTensor.dot(eigenvector)
			lambda_v = eigenvalue * eigenvector

			for i in range(0, 3):
				assert Sv[i] == pytest.approx(lambda_v[i])


def test___eq__():
	pc1 = ParticleCollection()
	pc2 = ParticleCollection()

	assert pc1 == pc1
	assert pc1 == pc2
	assert pc2 == pc2
	assert pc2 == pc1


	pc1.setUniformMass(1.0)

	assert pc1 == pc1
	assert not pc1 == pc2
	assert not pc2 == pc1
	assert pc2 == pc2

	pc2.setUniformMass(1.0)

	assert pc1 == pc1
	assert pc1 == pc2
	assert pc2 == pc2
	assert pc2 == pc1


	from MPCDAnalysis.Vector3DReal import Vector3DReal
	v1 = Vector3DReal(0, 1, 2)
	v2 = Vector3DReal(0.1, -1, 2.4)

	pc1.setPositionsAndVelocities([v1], [v2])
	assert pc1 == pc1
	assert not pc1 == pc2
	assert not pc2 == pc1
	assert pc2 == pc2

	pc2.setPositionsAndVelocities([v1], [v2])
	assert pc1 == pc1
	assert pc1 == pc2
	assert pc2 == pc2
	assert pc2 == pc1


	pc1.setPositionsAndVelocities([v1, v2], [v1, v2])
	assert pc1 == pc1
	assert not pc1 == pc2
	assert not pc2 == pc1
	assert pc2 == pc2

	pc2.setPositionsAndVelocities([v1], [v1])
	assert pc1 == pc1
	assert not pc1 == pc2
	assert not pc2 == pc1
	assert pc2 == pc2

	pc2.setPositionsAndVelocities([v1, v2], [v1, v2])
	assert pc1 == pc1
	assert pc1 == pc2
	assert pc2 == pc2
	assert pc2 == pc1


	pc2.setPositionsAndVelocities([v1, v2], [v1, v1])
	assert pc1 == pc1
	assert not pc1 == pc2
	assert not pc2 == pc1
	assert pc2 == pc2


	pc2.setPositionsAndVelocities([v1, v1], [v1, v2])
	assert pc1 == pc1
	assert not pc1 == pc2
	assert not pc2 == pc1
	assert pc2 == pc2


	pc1.setUniformMass(0.1)
	assert pc1 == pc1
	assert not pc1 == pc2
	assert not pc2 == pc1
	assert pc2 == pc2


def test___ne__():
	pc1 = ParticleCollection()
	pc2 = ParticleCollection()

	assert not pc1 != pc1
	assert not pc1 != pc2
	assert not pc2 != pc2
	assert not pc2 != pc1


	pc1.setUniformMass(1.0)

	assert not pc1 != pc1
	assert pc1 != pc2
	assert pc2 != pc1
	assert not pc2 != pc2

	pc2.setUniformMass(1.0)

	assert not pc1 != pc1
	assert not pc1 != pc2
	assert not pc2 != pc2
	assert not pc2 != pc1


	from MPCDAnalysis.Vector3DReal import Vector3DReal
	v1 = Vector3DReal(0, 1, 2)
	v2 = Vector3DReal(0.1, -1, 2.4)

	pc1.setPositionsAndVelocities([v1], [v2])
	assert not pc1 != pc1
	assert pc1 != pc2
	assert pc2 != pc1
	assert not pc2 != pc2

	pc2.setPositionsAndVelocities([v1], [v2])
	assert not pc1 != pc1
	assert not pc1 != pc2
	assert not pc2 != pc2
	assert not pc2 != pc1


	pc1.setPositionsAndVelocities([v1, v2], [v1, v2])
	assert pc1 == pc1
	assert not pc1 == pc2
	assert not pc2 == pc1
	assert pc2 == pc2

	pc2.setPositionsAndVelocities([v1], [v1])
	assert pc1 == pc1
	assert not pc1 == pc2
	assert not pc2 == pc1
	assert pc2 == pc2

	pc2.setPositionsAndVelocities([v1, v2], [v1, v2])
	assert pc1 == pc1
	assert pc1 == pc2
	assert pc2 == pc2
	assert pc2 == pc1


	pc2.setPositionsAndVelocities([v1, v2], [v1, v1])
	assert not pc1 != pc1
	assert pc1 != pc2
	assert pc2 != pc1
	assert not pc2 != pc2


	pc2.setPositionsAndVelocities([v1, v1], [v1, v2])
	assert not pc1 != pc1
	assert pc1 != pc2
	assert pc2 != pc1
	assert not pc2 != pc2


	pc1.setUniformMass(0.1)
	assert pc1 == pc1
	assert not pc1 == pc2
	assert not pc2 == pc1
	assert pc2 == pc2
