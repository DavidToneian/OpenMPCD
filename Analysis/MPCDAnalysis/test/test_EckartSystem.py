from MPCDAnalysis.EckartSystem import EckartSystem

def getParticles(setMass = True):
	import os.path
	vtfPath = os.path.dirname(os.path.abspath(__file__))
	vtfPath += "/data/test_EckartSystem_particles.vtf"

	from MPCDAnalysis.VTFSnapshotFile import VTFSnapshotFile
	vtf = VTFSnapshotFile(vtfPath)

	particles = vtf.readTimestep()

	from MPCDAnalysis.ParticleCollection import ParticleCollection
	assert isinstance(particles, ParticleCollection)

	if setMass:
		particles.setUniformMass(1.23)
	return particles


def test___init__():
	import pytest

	with pytest.raises(TypeError):
		EckartSystem([[0.0, 0.0, 0.0]])


	particles = getParticles(setMass = False)

	with pytest.raises(ValueError):
		EckartSystem(particles)

	particles.setUniformMass(1.23)
	eckartSystem = EckartSystem(particles)

	assert eckartSystem._referenceConfiguration != particles
	particles.shiftToCenterOfMassFrame()
	assert eckartSystem._referenceConfiguration == particles

	particles.setPositionsAndVelocities([], [])
	assert eckartSystem._referenceConfiguration != particles


def test_getParticleCount():
	particles = getParticles()
	eckartSystem = EckartSystem(particles)

	assert eckartSystem.getParticleCount() == particles.getParticleCount()
	assert eckartSystem.getParticleCount() == particles.getParticleCount()
	assert isinstance(eckartSystem.getParticleCount(), int)


def test_getReferencePosition():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import pytest


	particles = getParticles()
	eckartSystem = EckartSystem(particles)

	with pytest.raises(TypeError):
		eckartSystem.getReferencePosition(0.0)
	with pytest.raises(KeyError):
		eckartSystem.getReferencePosition(-1)
	with pytest.raises(KeyError):
		eckartSystem.getReferencePosition(eckartSystem.getParticleCount())
	with pytest.raises(KeyError):
		eckartSystem.getReferencePosition(eckartSystem.getParticleCount() + 1)

	particles.shiftToCenterOfMassFrame()
	for i in range(0, eckartSystem.getParticleCount()):
		assert eckartSystem.getReferencePosition(i) == particles.getPosition(i)
		assert eckartSystem.getReferencePosition(i) == particles.getPosition(i)
		assert isinstance(eckartSystem.getReferencePosition(i), Vector3DReal)


def test_getEckartVectors():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import pytest


	def getExpected(eckartSystem, particles):
		expected = [Vector3DReal(0, 0, 0) for _ in range(0, 3)]
		for p in range(0, eckartSystem.getParticleCount()):
			for i in range(0, 3):
				scale = particles.getMass(p)
				scale *= eckartSystem.getReferencePosition(p)[i]
				expected[i] += particles.getPosition(p) * scale
		return expected


	particles = getParticles()
	eckartSystem = EckartSystem(particles)

	with pytest.raises(TypeError):
		eckartSystem.getEckartVectors([[0.0, 0.0, 0.0]])

	particles.setUniformMass(0.67)
	with pytest.raises(ValueError):
		eckartSystem.getEckartVectors(particles)

	particles = getParticles()
	positions = particles.getPositions()
	velocities = particles.getVelocities()
	particles.setPositionsAndVelocities(positions[:3], velocities[:3])
	with pytest.raises(ValueError):
		eckartSystem.getEckartVectors(particles)

	particles = getParticles()

	assert eckartSystem.getEckartVectors(particles) \
	       == getExpected(eckartSystem, particles)
	assert eckartSystem.getEckartVectors(particles) \
	       == getExpected(eckartSystem, particles)

	particles.shiftToCenterOfMassFrame()
	assert eckartSystem.getEckartVectors(particles) \
	       == getExpected(eckartSystem, particles)
	assert eckartSystem.getEckartVectors(particles) \
	       == getExpected(eckartSystem, particles)


	import random
	positions = []
	velocities = []
	for _ in range(0, eckartSystem.getParticleCount()):
		r = [random.uniform(-3.0, 5.0) for _2 in range(0, 6)]
		positions.append(Vector3DReal(r[0], r[1], r[2]))
		velocities.append(Vector3DReal(r[3], r[4], r[5]))
	particles.setPositionsAndVelocities(positions, velocities)

	assert eckartSystem.getEckartVectors(particles) \
	       == getExpected(eckartSystem, particles)
	assert eckartSystem.getEckartVectors(particles) \
	       == getExpected(eckartSystem, particles)

	particles.shiftToCenterOfMassFrame()
	assert eckartSystem.getEckartVectors(particles) \
	       == getExpected(eckartSystem, particles)
	assert eckartSystem.getEckartVectors(particles) \
	       == getExpected(eckartSystem, particles)


	eckartVectors = eckartSystem.getEckartVectors(particles)
	assert isinstance(eckartVectors, list)
	for v in eckartVectors:
		assert isinstance(v, Vector3DReal)


def test_getEckartVectors_invariant_under_center_of_mass_translation():
	particles = getParticles()
	eckartSystem = EckartSystem(particles)
	eckartVectors = eckartSystem.getEckartVectors(particles)

	def check(particles):
		for _ in range(0, 2):
			ev = eckartSystem.getEckartVectors(particles)
			for i in range(0, 3):
				assert ev[i].isClose(eckartVectors[i])


	check(particles)

	particles.shiftToCenterOfMassFrame()
	check(particles)

	from MPCDAnalysis.Vector3DReal import Vector3DReal
	import random
	for _ in range(0, 5):
		r = [random.uniform(-5.0, 2.5) for _2 in range(0, 3)]
		offset = Vector3DReal(*r)
		positions = []
		for i in range(0, particles.getParticleCount()):
			positions.append(particles.getPosition(i) + offset)
		particles.setPositionsAndVelocities(positions, positions)
		check(particles)


def test_getGramMatrix():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import pytest


	def getExpected(eckartSystem, particles):
		eckartVectors = eckartSystem.getEckartVectors(particles)
		expected = []

		for i in range(0, 3):
			row = []
			for j in range(0, 3):
				row.append(eckartVectors[i].dot(eckartVectors[j]))
			expected.append(row)

		return expected


	particles = getParticles()
	eckartSystem = EckartSystem(particles)

	with pytest.raises(TypeError):
		eckartSystem.getGramMatrix([[0.0, 0.0, 0.0]])

	particles.setUniformMass(0.67)
	with pytest.raises(ValueError):
		eckartSystem.getGramMatrix(particles)

	particles = getParticles()
	positions = particles.getPositions()
	velocities = particles.getVelocities()
	particles.setPositionsAndVelocities(positions[:3], velocities[:3])
	with pytest.raises(ValueError):
		eckartSystem.getGramMatrix(particles)

	particles = getParticles()

	assert eckartSystem.getGramMatrix(particles) \
	       == getExpected(eckartSystem, particles)
	assert eckartSystem.getGramMatrix(particles) \
	       == getExpected(eckartSystem, particles)

	particles.shiftToCenterOfMassFrame()
	assert eckartSystem.getGramMatrix(particles) \
	       == getExpected(eckartSystem, particles)
	assert eckartSystem.getGramMatrix(particles) \
	       == getExpected(eckartSystem, particles)


	import random
	positions = []
	velocities = []
	for _ in range(0, eckartSystem.getParticleCount()):
		r = [random.uniform(-3.0, 5.0) for _2 in range(0, 6)]
		positions.append(Vector3DReal(r[0], r[1], r[2]))
		velocities.append(Vector3DReal(r[3], r[4], r[5]))
	particles.setPositionsAndVelocities(positions, velocities)

	assert eckartSystem.getGramMatrix(particles) \
	       == getExpected(eckartSystem, particles)
	assert eckartSystem.getGramMatrix(particles) \
	       == getExpected(eckartSystem, particles)

	particles.shiftToCenterOfMassFrame()
	assert eckartSystem.getGramMatrix(particles) \
	       == getExpected(eckartSystem, particles)
	assert eckartSystem.getGramMatrix(particles) \
	       == getExpected(eckartSystem, particles)


	gramMatrix = eckartSystem.getGramMatrix(particles)
	assert isinstance(gramMatrix, list)
	for v in gramMatrix:
		assert isinstance(v, list)
		for x in v:
			assert isinstance(x, float)


def test_getEckartFrame():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import pytest


	def check(eckartSystem, particles):
		import numpy

		ev = eckartSystem.getEckartVectors(particles)
		ev_plain = [[v.getX(), v.getY(), v.getZ()] for v in ev]
		left = numpy.column_stack(ev_plain)
		right = eckartSystem._getGramMatrixInverseSquareRoot(particles)
		expected = numpy.dot(left, right)
		assert expected.shape == (3, 3)

		for _ in range(0, 3):
			result = eckartSystem.getEckartFrame(particles)
			assert isinstance(result, list)
			assert len(result) == 3
			for v in result:
				assert isinstance(v, Vector3DReal)

			for i in range(0, 3):
				assert result[i].getX() == pytest.approx(expected[0][i])
				assert result[i].getY() == pytest.approx(expected[1][i])
				assert result[i].getZ() == pytest.approx(expected[2][i])


			nullVector = Vector3DReal(0, 0, 0)
			sumOfCrosses = nullVector
			for i in range(0, 3):
				sumOfCrosses += result[i].cross(ev[i])
			assert sumOfCrosses.isClose(nullVector)

	particles = getParticles()
	eckartSystem = EckartSystem(particles)

	with pytest.raises(TypeError):
		eckartSystem.getEckartFrame([[0.0, 0.0, 0.0]])

	particles.setUniformMass(0.67)
	with pytest.raises(ValueError):
		eckartSystem.getEckartFrame(particles)

	particles = getParticles()
	positions = particles.getPositions()
	velocities = particles.getVelocities()
	particles.setPositionsAndVelocities(positions[:3], velocities[:3])
	with pytest.raises(ValueError):
		eckartSystem.getEckartFrame(particles)

	particles = getParticles()
	check(eckartSystem, particles)

	particles.shiftToCenterOfMassFrame()
	check(eckartSystem, particles)


	import random
	positions = []
	velocities = []
	for _ in range(0, eckartSystem.getParticleCount()):
		r = [random.uniform(-3.0, 5.0) for _2 in range(0, 6)]
		positions.append(Vector3DReal(r[0], r[1], r[2]))
		velocities.append(Vector3DReal(r[3], r[4], r[5]))
	particles.setPositionsAndVelocities(positions, velocities)

	check(eckartSystem, particles)

	particles.shiftToCenterOfMassFrame()
	check(eckartSystem, particles)


def test_getEckartFrameEquilibriumPositions():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import pytest


	def check(eckartSystem, particles):
		ef = eckartSystem.getEckartFrame(particles)

		for _ in range(0, 3):
			result = \
				eckartSystem.getEckartFrameEquilibriumPositions(particles)

			assert isinstance(result, list)
			assert len(result) == eckartSystem.getParticleCount()

			for i in range(0, eckartSystem.getParticleCount()):
				assert isinstance(result[i], Vector3DReal)

				expected = Vector3DReal(0, 0, 0)

				for j in range(0, 3):
					expected += ef[j] * eckartSystem.getReferencePosition(i)[j]

				assert result[i].isClose(expected)


			nullVector = Vector3DReal(0, 0, 0)
			eckartCondition = nullVector
			centerOfMass = particles.getCenterOfMass()
			for i in range(0, eckartSystem.getParticleCount()):
				tmp1 = particles.getPosition(i) - centerOfMass
				tmp2 = result[i].cross(tmp1)
				eckartCondition += tmp2 * particles.getMass(i)
			assert eckartCondition.isClose(nullVector)


	particles = getParticles()
	eckartSystem = EckartSystem(particles)

	with pytest.raises(TypeError):
		eckartSystem.getEckartFrameEquilibriumPositions([[0.0, 0.0, 0.0]])

	particles.setUniformMass(0.67)
	with pytest.raises(ValueError):
		eckartSystem.getEckartFrameEquilibriumPositions(particles)

	particles = getParticles()
	positions = particles.getPositions()
	velocities = particles.getVelocities()
	particles.setPositionsAndVelocities(positions[:3], velocities[:3])
	with pytest.raises(ValueError):
		eckartSystem.getEckartFrameEquilibriumPositions(particles)

	particles = getParticles()
	check(eckartSystem, particles)

	particles.shiftToCenterOfMassFrame()
	check(eckartSystem, particles)


	import random
	positions = []
	velocities = []
	for _ in range(0, eckartSystem.getParticleCount()):
		r = [random.uniform(-3.0, 5.0) for _2 in range(0, 6)]
		positions.append(Vector3DReal(r[0], r[1], r[2]))
		velocities.append(Vector3DReal(r[3], r[4], r[5]))
	particles.setPositionsAndVelocities(positions, velocities)

	check(eckartSystem, particles)

	particles.shiftToCenterOfMassFrame()
	check(eckartSystem, particles)


def test_getEckartMomentOfInertiaTensor():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import copy
	import numpy
	import pytest


	def check(eckartSystem, particles):
		ef = eckartSystem.getEckartFrame(particles)

		for _ in range(0, 3):
			result = \
				eckartSystem.getEckartMomentOfInertiaTensor(particles)

			assert isinstance(result, numpy.ndarray)
			assert result.shape == (3, 3)

			comParticles = copy.deepcopy(particles)
			comParticles.shiftToCenterOfMassFrame()

			equilibriumPositions = \
				eckartSystem.getEckartFrameEquilibriumPositions(particles)

			expected = numpy.zeros((3, 3))
			for i in range(0, eckartSystem.getParticleCount()):
				m = comParticles.getMass(i)
				c = equilibriumPositions[i]
				r = comParticles.getPosition(i)
				tmp1 = r.dot(c)
				for j in range(0, 3):
					expected[j][j] += tmp1 * m

				for j in range(0, 3):
					for k in range(0, 3):
						expected[j][k] -= m * r[j] * c[k]

			assert numpy.allclose(result, expected)


	particles = getParticles()
	eckartSystem = EckartSystem(particles)

	with pytest.raises(TypeError):
		eckartSystem.getEckartMomentOfInertiaTensor([[0.0, 0.0, 0.0]])

	particles.setUniformMass(0.67)
	with pytest.raises(ValueError):
		eckartSystem.getEckartMomentOfInertiaTensor(particles)

	particles = getParticles()
	positions = particles.getPositions()
	velocities = particles.getVelocities()
	particles.setPositionsAndVelocities(positions[:3], velocities[:3])
	with pytest.raises(ValueError):
		eckartSystem.getEckartMomentOfInertiaTensor(particles)

	particles = getParticles()
	check(eckartSystem, particles)

	particles.shiftToCenterOfMassFrame()
	check(eckartSystem, particles)


	import random
	positions = []
	velocities = []
	for _ in range(0, eckartSystem.getParticleCount()):
		r = [random.uniform(-3.0, 5.0) for _2 in range(0, 6)]
		positions.append(Vector3DReal(r[0], r[1], r[2]))
		velocities.append(Vector3DReal(r[3], r[4], r[5]))
	particles.setPositionsAndVelocities(positions, velocities)

	check(eckartSystem, particles)

	particles.shiftToCenterOfMassFrame()
	check(eckartSystem, particles)


def test_getEckartAngularVelocityVector():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import copy
	import numpy
	import pytest


	def check(eckartSystem, particles):
		import scipy.linalg

		momentOfInertia = eckartSystem.getEckartMomentOfInertiaTensor(particles)
		invertedMomentOfInertia = scipy.linalg.inv(momentOfInertia)

		closeToUnity = numpy.dot(momentOfInertia, invertedMomentOfInertia)
		assert numpy.allclose(closeToUnity, numpy.identity(3))

		for _ in range(0, 3):
			result = \
				eckartSystem.getEckartAngularVelocityVector(particles)

			assert isinstance(result, Vector3DReal)

			comParticles = copy.deepcopy(particles)
			comParticles.shiftToCenterOfMassFrame()

			equilibriumPositions = \
				eckartSystem.getEckartFrameEquilibriumPositions(particles)

			tmp = Vector3DReal(0, 0, 0)
			for i in range(0, eckartSystem.getParticleCount()):
				m = comParticles.getMass(i)
				c = equilibriumPositions[i]
				v = comParticles.getVelocity(i)
				tmp += c.cross(v) * m

			numpyVector = [tmp.getX(), tmp.getY(), tmp.getZ()]
			npExpected = numpy.dot(invertedMomentOfInertia, numpyVector)
			expected = Vector3DReal(npExpected[0], npExpected[1], npExpected[2])
			assert result.isClose(expected)


	particles = getParticles()
	eckartSystem = EckartSystem(particles)

	with pytest.raises(TypeError):
		eckartSystem.getEckartAngularVelocityVector([[0.0, 0.0, 0.0]])

	particles.setUniformMass(0.67)
	with pytest.raises(ValueError):
		eckartSystem.getEckartAngularVelocityVector(particles)

	particles = getParticles()
	positions = particles.getPositions()
	velocities = particles.getVelocities()
	particles.setPositionsAndVelocities(positions[:3], velocities[:3])
	with pytest.raises(ValueError):
		eckartSystem.getEckartAngularVelocityVector(particles)

	particles = getParticles()
	check(eckartSystem, particles)

	particles.shiftToCenterOfMassFrame()
	check(eckartSystem, particles)


	import random
	positions = []
	velocities = []
	for _ in range(0, eckartSystem.getParticleCount()):
		r = [random.uniform(-3.0, 5.0) for _2 in range(0, 6)]
		positions.append(Vector3DReal(r[0], r[1], r[2]))
		velocities.append(Vector3DReal(r[3], r[4], r[5]))
	particles.setPositionsAndVelocities(positions, velocities)

	check(eckartSystem, particles)

	particles.shiftToCenterOfMassFrame()
	check(eckartSystem, particles)


def test__getGramMatrixInverseSquareRoot():
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import pytest

	def check(eckartSystem, particles):
		import numpy
		for _ in range(0, 3):
			gramMatrix = eckartSystem.getGramMatrix(particles)
			result = eckartSystem._getGramMatrixInverseSquareRoot(particles)
			inverse = numpy.dot(result, result)
			unity = numpy.dot(inverse, gramMatrix)
			assert numpy.allclose(unity, numpy.identity(3))

	particles = getParticles()
	eckartSystem = EckartSystem(particles)

	with pytest.raises(TypeError):
		eckartSystem.getGramMatrix([[0.0, 0.0, 0.0]])

	particles.setUniformMass(0.67)
	with pytest.raises(ValueError):
		eckartSystem.getGramMatrix(particles)

	particles = getParticles()
	positions = particles.getPositions()
	velocities = particles.getVelocities()
	particles.setPositionsAndVelocities(positions[:3], velocities[:3])
	with pytest.raises(ValueError):
		eckartSystem.getGramMatrix(particles)

	particles = getParticles()

	check(eckartSystem, particles)

	particles.shiftToCenterOfMassFrame()
	check(eckartSystem, particles)


	import random
	positions = []
	velocities = []
	for _ in range(0, eckartSystem.getParticleCount()):
		r = [random.uniform(-3.0, 5.0) for _2 in range(0, 6)]
		positions.append(Vector3DReal(r[0], r[1], r[2]))
		velocities.append(Vector3DReal(r[3], r[4], r[5]))
	particles.setPositionsAndVelocities(positions, velocities)

	check(eckartSystem, particles)

	particles.shiftToCenterOfMassFrame()
	check(eckartSystem, particles)


	import numpy
	matrix = eckartSystem._getGramMatrixInverseSquareRoot(particles)
	assert isinstance(matrix, numpy.ndarray)
	assert matrix.shape == (3, 3)
