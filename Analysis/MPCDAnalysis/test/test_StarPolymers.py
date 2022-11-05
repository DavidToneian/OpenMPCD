def getConfig():
	from MPCDAnalysis.Configuration import Configuration

	config = """
	structure:
	{
		starCount = 1
		armCountPerStar = 5
		armParticlesPerArm = 3
		hasMagneticParticles = true
		particleMass = 4.5
	}

	interactionParameters:
	{
		epsilon_core = 1.0
		epsilon_arm = 1.1
		epsilon_magnetic = 1.2

		sigma_core = 3.05
		sigma_arm = 3.06
		sigma_magnetic = 3.07

		D_core = 2.15
		D_arm = 0.01
		D_magnetic = 1.01

		magneticPrefactor = 100.7

		dipoleOrientation = [1.0, 0.0, 0.0]
	}
	"""

	return Configuration(config)


def getParticles(filename = "test_StarPolymers_particles.vtf"):
	from MPCDAnalysis.ParticleCollection import ParticleCollection
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	import os.path
	particlePath = os.path.dirname(os.path.abspath(__file__))
	particlePath += "/data/" + filename

	positions = []
	velocities = []
	with open(particlePath, "r") as vtf:
		for line in vtf:
			if line.startswith("atom"):
				continue
			if line.startswith("bond"):
				continue
			if line.startswith("timestep"):
				continue
			components = line.split()
			pos = [float(x) for x in components[0:3]]
			vel = [float(x) for x in components[3:6]]

			positions.append(Vector3DReal(pos))
			velocities.append(Vector3DReal(vel))

	ret = ParticleCollection()
	ret.setPositionsAndVelocities(positions, velocities)
	return ret


from MPCDAnalysis.StarPolymers import StarPolymers

def test_constructor():
	import pytest

	with pytest.raises(NotImplementedError):
		config = getConfig()
		config["structure.starCount"] = 2
		StarPolymers(config)


	with pytest.raises(TypeError):
		StarPolymers(0)

	with pytest.raises(TypeError):
		StarPolymers({"structure.starCount": 1})

	with pytest.raises(TypeError):
		StarPolymers(None)

	config = getConfig()
	sp = StarPolymers(config)

	assert sp.getArmCountPerStar() == 5
	config["structure.armCountPerStar"] = 2
	assert sp.getArmCountPerStar() == 5

def test_getStarCount():
	sp = StarPolymers(getConfig())

	assert sp.getStarCount() == 1
	assert sp.getStarCount() == 1


def test_getArmCountPerStar():
	sp = StarPolymers(getConfig())

	assert sp.getArmCountPerStar() == 5
	assert sp.getArmCountPerStar() == 5


def test_getArmParticlesPerArm():
	sp = StarPolymers(getConfig())

	assert sp.getArmParticlesPerArm() == 3
	assert sp.getArmParticlesPerArm() == 3


def test_getTotalParticleCountPerArm():
	sp = StarPolymers(getConfig())

	assert sp.getTotalParticleCountPerArm() == 3 + 1
	assert sp.getTotalParticleCountPerArm() == 3 + 1


def test_hasMagneticParticles():
	sp = StarPolymers(getConfig())

	assert sp.hasMagneticParticles() == True
	assert sp.hasMagneticParticles() == True


def test_getParticleMass():
	sp = StarPolymers(getConfig())

	assert sp.getParticleMass() == 4.5
	assert sp.getParticleMass() == 4.5


def test_getTotalParticleCountPerStar():
	sp = StarPolymers(getConfig())

	assert sp.getTotalParticleCountPerStar() == 1 + 5 * (3 + 1)


def test_getTotalParticleCount():
	sp = StarPolymers(getConfig())

	expected = sp.getStarCount() * sp.getTotalParticleCountPerStar()
	assert sp.getTotalParticleCount() == expected
	assert sp.getTotalParticleCount() == expected


def test_getParticleType():
	import pytest

	sp = StarPolymers(getConfig())

	assert sp.getTotalParticleCount() == 21

	with pytest.raises(IndexError):
		sp.getParticleType(-1)

	with pytest.raises(IndexError):
		sp.getParticleType(sp.getTotalParticleCount())

	with pytest.raises(IndexError):
		sp.getParticleType(sp.getTotalParticleCount() + 1)

	with pytest.raises(TypeError):
		sp.getParticleType(0.0)


	for _ in range(0, 3): #repeat to check caching
		assert sp.getParticleType(0) == "Core"

		assert sp.getParticleType(1) == "Arm"
		assert sp.getParticleType(2) == "Arm"
		assert sp.getParticleType(3) == "Arm"
		assert sp.getParticleType(4) == "Magnetic"

		assert sp.getParticleType(5) == "Arm"
		assert sp.getParticleType(6) == "Arm"
		assert sp.getParticleType(7) == "Arm"
		assert sp.getParticleType(8) == "Magnetic"

		assert sp.getParticleType(9) == "Arm"
		assert sp.getParticleType(10) == "Arm"
		assert sp.getParticleType(11) == "Arm"
		assert sp.getParticleType(12) == "Magnetic"

		assert sp.getParticleType(13) == "Arm"
		assert sp.getParticleType(14) == "Arm"
		assert sp.getParticleType(15) == "Arm"
		assert sp.getParticleType(16) == "Magnetic"

		assert sp.getParticleType(17) == "Arm"
		assert sp.getParticleType(18) == "Arm"
		assert sp.getParticleType(19) == "Arm"
		assert sp.getParticleType(20) == "Magnetic"


def test_getParticleStructureIndices():
	import pytest

	sp = StarPolymers(getConfig())

	assert sp.getTotalParticleCount() == 21

	with pytest.raises(IndexError):
		sp.getParticleStructureIndices(-1)

	with pytest.raises(IndexError):
		sp.getParticleStructureIndices(sp.getTotalParticleCount())

	with pytest.raises(IndexError):
		sp.getParticleStructureIndices(sp.getTotalParticleCount() + 1)

	with pytest.raises(TypeError):
		sp.getParticleStructureIndices(0.0)

	for _ in range(0, 3): #repeat to check caching
		assert sp.getParticleStructureIndices(0) == [0, None, None]

		assert sp.getParticleStructureIndices(1) == [0, 0, 0]
		assert sp.getParticleStructureIndices(2) == [0, 0, 1]
		assert sp.getParticleStructureIndices(3) == [0, 0, 2]
		assert sp.getParticleStructureIndices(4) == [0, 0, None]

		assert sp.getParticleStructureIndices(5) == [0, 1, 0]
		assert sp.getParticleStructureIndices(6) == [0, 1, 1]
		assert sp.getParticleStructureIndices(7) == [0, 1, 2]
		assert sp.getParticleStructureIndices(8) == [0, 1, None]

		assert sp.getParticleStructureIndices(9) == [0, 2, 0]
		assert sp.getParticleStructureIndices(10) == [0, 2, 1]
		assert sp.getParticleStructureIndices(11) == [0, 2, 2]
		assert sp.getParticleStructureIndices(12) == [0, 2, None]

		assert sp.getParticleStructureIndices(13) == [0, 3, 0]
		assert sp.getParticleStructureIndices(14) == [0, 3, 1]
		assert sp.getParticleStructureIndices(15) == [0, 3, 2]
		assert sp.getParticleStructureIndices(16) == [0, 3, None]

		assert sp.getParticleStructureIndices(17) == [0, 4, 0]
		assert sp.getParticleStructureIndices(18) == [0, 4, 1]
		assert sp.getParticleStructureIndices(19) == [0, 4, 2]
		assert sp.getParticleStructureIndices(20) == [0, 4, None]


def test_particlesAreBonded():
	import pytest

	sp = StarPolymers(getConfig())

	assert sp.getTotalParticleCount() == 21

	with pytest.raises(IndexError):
		sp.particlesAreBonded(-1, 0)

	with pytest.raises(IndexError):
		sp.particlesAreBonded(0, -1)

	with pytest.raises(IndexError):
		sp.particlesAreBonded(sp.getTotalParticleCount(), 0)

	with pytest.raises(IndexError):
		sp.particlesAreBonded(0, sp.getTotalParticleCount())

	with pytest.raises(IndexError):
		sp.particlesAreBonded(sp.getTotalParticleCount() + 1, 0)

	with pytest.raises(IndexError):
		sp.particlesAreBonded(0, sp.getTotalParticleCount() + 1)

	with pytest.raises(TypeError):
		sp.particlesAreBonded(0.0, 0)

	with pytest.raises(TypeError):
		sp.particlesAreBonded(0, 0.0)


	for pID1 in range(0, sp.getTotalParticleCount()):
		assert sp.particlesAreBonded(pID1, pID1) == False

		for pID2 in range(pID1 + 1, sp.getTotalParticleCount()):
			assert sp.particlesAreBonded(pID1, pID2) \
			       == sp.particlesAreBonded(pID2, pID1)

			indices1 = sp.getParticleStructureIndices(pID1)
			indices2 = sp.getParticleStructureIndices(pID2)

			if indices1[0] != indices2[0]:
				assert sp.particlesAreBonded(pID1, pID2) == False
				continue

			if sp.getParticleType(pID1) == "Core":
				assert sp.hasMagneticParticles()
				bonded = indices2[2] == 0
				assert sp.particlesAreBonded(pID1, pID2) == bonded
			elif sp.getParticleType(pID1) == "Magnetic":
				#since pID2 > pID1, pID2 must be on a different arm:
				assert sp.particlesAreBonded(pID1, pID2) == False
			else:
				if indices1[1] == indices2[1]:
					bonded = pID2 - pID1 == 1 #using pID2 > pID1
					assert sp.particlesAreBonded(pID1, pID2) == bonded
				else:
					assert sp.particlesAreBonded(pID1, pID2) == False


def test_setParticles():
	from MPCDAnalysis.ParticleCollection import ParticleCollection
	from MPCDAnalysis.Vector3DReal import Vector3DReal
	import pytest

	sp = StarPolymers(getConfig())

	particleCount = sp.getTotalParticleCount()
	assert particleCount == 21

	with pytest.raises(TypeError):
		sp.setParticles(None)

	with pytest.raises(TypeError):
		sp.setParticles([[0.0, 1.0, 2.0] for _ in range(0, 21)])

	with pytest.raises(ValueError):
		particles = ParticleCollection()
		positions = [Vector3DReal(i, 0, 0) for i in range(0, particleCount - 1)]
		velocities = positions
		particles.setPositionsAndVelocities(positions, velocities)
		sp.setParticles(particles)


	particles = getParticles()
	sp.setParticles(particles)

	assert sp._particles is particles


def test_getMagneticClusters_getMagneticClusterCount():
	import pytest

	particles = getParticles()

	def test_raiseBecauseNoMagnets():
		config = getConfig()
		config["structure.hasMagneticParticles"] = False
		config["structure.armParticlesPerArm"] = \
			config["structure.armParticlesPerArm"] + 1
		mysp = StarPolymers(config)

		with pytest.raises(ValueError):
			mysp.getMagneticClusters(2.5)

		with pytest.raises(ValueError):
			mysp.getMagneticClusterCount(2.5)
	test_raiseBecauseNoMagnets()


	sp = StarPolymers(getConfig())
	assert sp.getTotalParticleCount() == 21

	with pytest.raises(ValueError):
		sp.getMagneticClusters(2.5)
	with pytest.raises(ValueError):
		sp.getMagneticClusterCount(2.5)


	sp.setParticles(particles)


	with pytest.raises(TypeError):
		sp.getMagneticClusters([1])
	with pytest.raises(TypeError):
		sp.getMagneticClusterCount([1])

	with pytest.raises(ValueError):
		sp.getMagneticClusters(-1)
	with pytest.raises(ValueError):
		sp.getMagneticClusterCount(-1)

	with pytest.raises(ValueError):
		sp.getMagneticClusters(-1.0)
	with pytest.raises(ValueError):
		sp.getMagneticClusterCount(-1.0)


	magnets = []
	for i in [4, 8, 12, 16, 20]:
		assert sp.getParticleType(i) == "Magnetic"
		magnets.append(particles.getPosition(i))

	def indexByPosition(position):
		for index, pos in enumerate(magnets):
			if pos is position:
				return index
		raise RuntimeError()


	expectedClustersCollection = \
		{
			0: [ [0], [1], [2], [3], [4] ],
			0.8: [ [0], [1], [2], [3], [4] ],
			0.9: [ [0, 3], [1], [2], [4] ],
			3: [ [0, 3], [1], [2], [4] ],
			3.3: [ [0, 3, 4], [1], [2] ],
			7.8: [ [0, 3, 4], [1, 2] ],
			7.9: [ [0, 1, 2, 3, 4] ],
			100: [ [0, 1, 2, 3, 4] ]
		}


	for distance, expectedClusters in expectedClustersCollection.items():
		assert len(expectedClusters) == sp.getMagneticClusterCount(distance)

		clusterSet = set()
		for cluster in sp.getMagneticClusters(distance):
			indices = [indexByPosition(pos) for pos in cluster]
			clusterSet.add(frozenset(indices))

		expectedClusterSet = set()
		for cluster in expectedClusters:
			expectedClusterSet.add(frozenset(cluster))

		assert clusterSet == expectedClusterSet


def test_getWCAPotentialParameterEpsilon():
	import pytest

	config = getConfig()
	config_nomagnetic = getConfig()
	config_nomagnetic["structure.hasMagneticParticles"] = False
	sp = StarPolymers(config)
	sp_nomagnetic = StarPolymers(config_nomagnetic)

	with pytest.raises(TypeError):
		sp.getWCAPotentialParameterEpsilon(0, "Core")
	with pytest.raises(TypeError):
		sp.getWCAPotentialParameterEpsilon("Core", 0)

	with pytest.raises(ValueError):
		sp.getWCAPotentialParameterEpsilon("Core", "Foo")
	with pytest.raises(ValueError):
		sp.getWCAPotentialParameterEpsilon("Foo", "Core")
	with pytest.raises(ValueError):
		sp_nomagnetic.getWCAPotentialParameterEpsilon("Core", "Magnetic")
	with pytest.raises(ValueError):
		sp_nomagnetic.getWCAPotentialParameterEpsilon("Magnetic", "Core")


	epsilon_C = config["interactionParameters.epsilon_core"]
	epsilon_A = config["interactionParameters.epsilon_arm"]
	epsilon_M = config["interactionParameters.epsilon_magnetic"]

	epsilon_CC = epsilon_C
	epsilon_AA = epsilon_A
	epsilon_MM = epsilon_M

	import math
	epsilon_CA = math.sqrt(epsilon_C * epsilon_A)
	epsilon_CM = math.sqrt(epsilon_C * epsilon_M)
	epsilon_AM = math.sqrt(epsilon_A * epsilon_M)

	assert sp.getWCAPotentialParameterEpsilon("Core", "Core") == epsilon_CC
	assert sp.getWCAPotentialParameterEpsilon("Core", "Arm") == epsilon_CA
	assert sp.getWCAPotentialParameterEpsilon("Core", "Magnetic") == epsilon_CM

	assert sp.getWCAPotentialParameterEpsilon("Arm", "Core") == epsilon_CA
	assert sp.getWCAPotentialParameterEpsilon("Arm", "Arm") == epsilon_AA
	assert sp.getWCAPotentialParameterEpsilon("Arm", "Magnetic") == epsilon_AM

	assert sp.getWCAPotentialParameterEpsilon("Magnetic", "Core") == epsilon_CM
	assert sp.getWCAPotentialParameterEpsilon("Magnetic", "Arm") == epsilon_AM
	assert sp.getWCAPotentialParameterEpsilon("Magnetic", "Magnetic") == \
	       epsilon_MM


def test_getWCAPotentialParameterSigma():
	import pytest

	config = getConfig()
	config_nomagnetic = getConfig()
	config_nomagnetic["structure.hasMagneticParticles"] = False
	sp = StarPolymers(config)
	sp_nomagnetic = StarPolymers(config_nomagnetic)

	with pytest.raises(TypeError):
		sp.getWCAPotentialParameterSigma(0, "Core")
	with pytest.raises(TypeError):
		sp.getWCAPotentialParameterSigma("Core", 0)

	with pytest.raises(ValueError):
		sp.getWCAPotentialParameterSigma("Core", "Foo")
	with pytest.raises(ValueError):
		sp.getWCAPotentialParameterSigma("Foo", "Core")
	with pytest.raises(ValueError):
		sp_nomagnetic.getWCAPotentialParameterSigma("Core", "Magnetic")
	with pytest.raises(ValueError):
		sp_nomagnetic.getWCAPotentialParameterSigma("Magnetic", "Core")


	sigma_C = config["interactionParameters.sigma_core"]
	sigma_A = config["interactionParameters.sigma_arm"]
	sigma_M = config["interactionParameters.sigma_magnetic"]

	sigma_CC = sigma_C
	sigma_AA = sigma_A
	sigma_MM = sigma_M

	sigma_CA = (sigma_C + sigma_A) / 2.0
	sigma_CM = (sigma_C + sigma_M) / 2.0
	sigma_AM = (sigma_A + sigma_M) / 2.0

	assert sp.getWCAPotentialParameterSigma("Core", "Core") == sigma_CC
	assert sp.getWCAPotentialParameterSigma("Core", "Arm") == sigma_CA
	assert sp.getWCAPotentialParameterSigma("Core", "Magnetic") == sigma_CM

	assert sp.getWCAPotentialParameterSigma("Arm", "Core") == sigma_CA
	assert sp.getWCAPotentialParameterSigma("Arm", "Arm") == sigma_AA
	assert sp.getWCAPotentialParameterSigma("Arm", "Magnetic") == sigma_AM

	assert sp.getWCAPotentialParameterSigma("Magnetic", "Core") == sigma_CM
	assert sp.getWCAPotentialParameterSigma("Magnetic", "Arm") == sigma_AM
	assert sp.getWCAPotentialParameterSigma("Magnetic", "Magnetic") == sigma_MM


def test_getWCAPotentialParameterD():
	import pytest

	config = getConfig()
	config_nomagnetic = getConfig()
	config_nomagnetic["structure.hasMagneticParticles"] = False
	sp = StarPolymers(config)
	sp_nomagnetic = StarPolymers(config_nomagnetic)

	with pytest.raises(TypeError):
		sp.getWCAPotentialParameterD(0, "Core")
	with pytest.raises(TypeError):
		sp.getWCAPotentialParameterD("Core", 0)

	with pytest.raises(ValueError):
		sp.getWCAPotentialParameterD("Core", "Foo")
	with pytest.raises(ValueError):
		sp.getWCAPotentialParameterD("Foo", "Core")
	with pytest.raises(ValueError):
		sp_nomagnetic.getWCAPotentialParameterD("Core", "Magnetic")
	with pytest.raises(ValueError):
		sp_nomagnetic.getWCAPotentialParameterD("Magnetic", "Core")


	D_C = config["interactionParameters.D_core"]
	D_A = config["interactionParameters.D_arm"]
	D_M = config["interactionParameters.D_magnetic"]

	D_CC = D_C
	D_AA = D_A
	D_MM = D_M

	D_CA = (D_C + D_A) / 2.0
	D_CM = (D_C + D_M) / 2.0
	D_AM = (D_A + D_M) / 2.0

	assert sp.getWCAPotentialParameterD("Core", "Core") == D_CC
	assert sp.getWCAPotentialParameterD("Core", "Arm") == D_CA
	assert sp.getWCAPotentialParameterD("Core", "Magnetic") == D_CM

	assert sp.getWCAPotentialParameterD("Arm", "Core") == D_CA
	assert sp.getWCAPotentialParameterD("Arm", "Arm") == D_AA
	assert sp.getWCAPotentialParameterD("Arm", "Magnetic") == D_AM

	assert sp.getWCAPotentialParameterD("Magnetic", "Core") == D_CM
	assert sp.getWCAPotentialParameterD("Magnetic", "Arm") == D_AM
	assert sp.getWCAPotentialParameterD("Magnetic", "Magnetic") == D_MM


def test_getWCAPotential():
	import pytest

	config = getConfig()
	config_nomagnetic = getConfig()
	config_nomagnetic["structure.hasMagneticParticles"] = False
	sp = StarPolymers(config)
	sp_nomagnetic = StarPolymers(config_nomagnetic)

	with pytest.raises(TypeError):
		sp.getWCAPotential(0, "Core")
	with pytest.raises(TypeError):
		sp.getWCAPotential("Core", 0)

	with pytest.raises(ValueError):
		sp.getWCAPotential("Core", "Foo")
	with pytest.raises(ValueError):
		sp.getWCAPotential("Foo", "Core")
	with pytest.raises(ValueError):
		sp_nomagnetic.getWCAPotential("Core", "Magnetic")
	with pytest.raises(ValueError):
		sp_nomagnetic.getWCAPotential("Magnetic", "Core")


	from MPCDAnalysis.PairPotentials.WeeksChandlerAndersen_DistanceOffset \
		import WeeksChandlerAndersen_DistanceOffset as WCA
	types = ["Core", "Arm", "Magnetic"]
	for type1 in types:
		for type2 in types:
			epsilon = sp.getWCAPotentialParameterEpsilon(type1, type2)
			sigma = sp.getWCAPotentialParameterSigma(type1, type2)
			d = sp.getWCAPotentialParameterD(type1, type2)

			wca = sp.getWCAPotential(type1, type2)

			assert isinstance(wca, WCA)
			assert wca.getEpsilon() == epsilon
			assert wca.getSigma() == sigma
			assert wca.getD() == d

			#check that caching works:
			wca = sp.getWCAPotential(type1, type2)

			assert isinstance(wca, WCA)
			assert wca.getEpsilon() == epsilon
			assert wca.getSigma() == sigma
			assert wca.getD() == d


def test_getFENEPotential():
	import pytest

	config = getConfig()
	config_nomagnetic = getConfig()
	config_nomagnetic["structure.hasMagneticParticles"] = False
	sp = StarPolymers(config)
	sp_nomagnetic = StarPolymers(config_nomagnetic)

	with pytest.raises(TypeError):
		sp.getFENEPotential(0, "Core")
	with pytest.raises(TypeError):
		sp.getFENEPotential("Core", 0)

	with pytest.raises(ValueError):
		sp.getFENEPotential("Core", "Foo")
	with pytest.raises(ValueError):
		sp.getFENEPotential("Foo", "Core")
	with pytest.raises(ValueError):
		sp_nomagnetic.getFENEPotential("Core", "Magnetic")
	with pytest.raises(ValueError):
		sp_nomagnetic.getFENEPotential("Magnetic", "Core")


	from MPCDAnalysis.PairPotentials.FENE import FENE

	types = ["Core", "Arm", "Magnetic"]
	for type1 in types:
		for type2 in types:
			epsilon = sp.getWCAPotentialParameterEpsilon(type1, type2)
			sigma = sp.getWCAPotentialParameterSigma(type1, type2)
			d = sp.getWCAPotentialParameterD(type1, type2)

			fene = sp.getFENEPotential(type1, type2)

			assert isinstance(fene, FENE)
			assert fene.getK() == pytest.approx(30 * epsilon * sigma ** -2)
			assert fene.get_l_0() == d
			assert fene.getR() == 1.5 * sigma

			#check that caching works:
			fene = sp.getFENEPotential(type1, type2)

			assert isinstance(fene, FENE)
			assert fene.getK() == pytest.approx(30 * epsilon * sigma ** -2)
			assert fene.get_l_0() == d
			assert fene.getR() == 1.5 * sigma


def test_getMagneticPotential():
	import pytest

	config = getConfig()
	config_nomagnetic = getConfig()
	config_nomagnetic["structure.hasMagneticParticles"] = False
	sp = StarPolymers(config)
	sp_nomagnetic = StarPolymers(config_nomagnetic)

	with pytest.raises(RuntimeError):
		sp_nomagnetic.getMagneticPotential()


	from MPCDAnalysis.PairPotentials.MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles \
		import MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles \
		as Potential
	from MPCDAnalysis.Vector3DReal import Vector3DReal

	prefactor = config["interactionParameters.magneticPrefactor"]
	orientationX = config["interactionParameters.dipoleOrientation.[0]"]
	orientationY = config["interactionParameters.dipoleOrientation.[1]"]
	orientationZ = config["interactionParameters.dipoleOrientation.[2]"]
	orientation = Vector3DReal(orientationX, orientationY, orientationZ)

	potential = sp.getMagneticPotential()

	assert isinstance(potential, Potential)
	assert potential.getPrefactor() == prefactor
	assert potential.getOrientation() == orientation

	#check that caching works:
	potential = sp.getMagneticPotential()

	assert isinstance(potential, Potential)
	assert potential.getPrefactor() == prefactor
	assert potential.getOrientation() == orientation


def test_getPotentialEnergy():
	import pytest

	particles = getParticles("test_StarPolymers_particles_2.vtf")

	sp = StarPolymers(getConfig())
	assert sp.getStarCount() == 1
	assert sp.getTotalParticleCount() == 21

	with pytest.raises(ValueError):
		sp.getPotentialEnergy()

	wca_CA = sp.getWCAPotential("Core", "Arm")
	wca_CM = sp.getWCAPotential("Core", "Magnetic")
	wca_AA = sp.getWCAPotential("Arm", "Arm")
	wca_AM = sp.getWCAPotential("Arm", "Magnetic")
	wca_MM = sp.getWCAPotential("Magnetic", "Magnetic")

	fene_CA = sp.getFENEPotential("Core", "Arm")
	fene_AA = sp.getFENEPotential("Arm", "Arm")
	fene_AM = sp.getFENEPotential("Arm", "Magnetic")

	magnetic = sp.getMagneticPotential()

	arms = [ [1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19]]
	magneticParticles = [4, 8, 12, 16, 20]

	for arm in arms:
		for p in arm:
			assert sp.getParticleType(p) == "Arm"
	for p in magneticParticles:
		assert sp.getParticleType(p) == "Magnetic"


	sp.setParticles(particles)

	expected = 0.0

	pos_C = particles.getPosition(0)
	for arm in arms:
		for i, p in enumerate(arm):
			r = pos_C - particles.getPosition(p)

			expected += wca_CA.getPotential(r)
			if i == 0:
				expected += fene_CA.getPotential(r)
	for p in magneticParticles:
		r = pos_C - particles.getPosition(p)
		expected += wca_CM.getPotential(r)


	for arm1 in arms:
		for arm2 in arms:
			for p1 in arm1:
				pos1 = particles.getPosition(p1)
				for p2 in arm2:
					if p2 <= p1:
						continue
					r = pos1 - particles.getPosition(p2)

					expected += wca_AA.getPotential(r)

					if p2 - p1 == 1:
						expected += fene_AA.getPotential(r)

	for arm in arms:
		for p1 in arm:
			pos1 = particles.getPosition(p1)
			for p2 in magneticParticles:
				r = pos1 - particles.getPosition(p2)

				expected += wca_AM.getPotential(r)


	for p1 in magneticParticles:
		pos1 = particles.getPosition(p1)

		bonded = particles.getPosition(p1 - 1)
		expected += fene_AM.getPotential(pos1 - bonded)

		for p2 in magneticParticles:
			if p2 <= p1:
				continue
			r = pos1 - particles.getPosition(p2)
			expected += wca_MM.getPotential(r)
			expected += magnetic.getPotential(r)

	assert sp.getPotentialEnergy() == expected

	#check that caching works:
	assert sp.getPotentialEnergy() == expected
