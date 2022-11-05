from .ParticleCollection import ParticleCollection

class StarPolymers:
	"""
	Representation of a collection of star polymers, as described in
	`OpenMPCD::CUDA::MPCSolute::StarPolymers`.
	"""

	def __init__(self, config):
		"""
		The constructor.

		@throw NotImplementedError
		       Throws if more than one star is configured.

		@param[in] config
		           An instance of `Configuration` that contains the star polymer
		           configuration as its root element. A copy of this instance is
		           stored, rather than a reference to the given instance.
		"""

		from .Configuration import Configuration
		if not isinstance(config, Configuration):
			raise TypeError()

		import copy
		self._config = copy.deepcopy(config)

		self._particles = None

		self._caches = \
			{
				"getParticleType": {},
				"getParticleStructureIndices": {},
				"particlesAreBonded": {},
				"getWCAPotential": {},
				"getFENEPotential": {},
				"getMagneticPotential": None,
			}

		if self.getStarCount() != 1:
			raise NotImplementedError("Unimplemented")


	def getStarCount(self):
		"""
		Returns the number of stars configured.
		"""

		if "getStarCount" not in self._caches:
			self._caches["getStarCount"] = self._config["structure.starCount"]

		return self._caches["getStarCount"]


	def getArmCountPerStar(self):
		"""
		Returns the number of arms configured per star.
		"""

		if "getArmCountPerStar" not in self._caches:
			self._caches["getArmCountPerStar"] = \
				self._config["structure.armCountPerStar"]

		return self._caches["getArmCountPerStar"]


	def getArmParticlesPerArm(self):
		"""
		Returns the number of arm particles configured per arm.
		"""

		if "getArmParticlesPerArm" not in self._caches:
			self._caches["getArmParticlesPerArm"] = \
				self._config["structure.armParticlesPerArm"]

		return self._caches["getArmParticlesPerArm"]


	def getTotalParticleCountPerArm(self):
		"""
		Returns the total number of particles per arm.
		"""

		if "getTotalParticleCountPerArm" in self._caches:
			return self._caches["getTotalParticleCountPerArm"]

		ret = self.getArmParticlesPerArm()
		if self.hasMagneticParticles():
			ret += 1

		self._caches["getTotalParticleCountPerArm"] = ret
		return ret


	def hasMagneticParticles(self):
		"""
		Returns whether arms end in a magnetic particle.
		"""

		if "hasMagneticParticles" not in self._caches:
			self._caches["hasMagneticParticles"] = \
				self._config["structure.hasMagneticParticles"]

		return self._caches["hasMagneticParticles"]


	def getParticleMass(self):
		"""
		Returns the mass of a particle.
		"""

		if "getParticleMass" not in self._caches:
			self._caches["getParticleMass"] = \
				self._config["structure.particleMass"]

		return self._caches["getParticleMass"]


	def getTotalParticleCountPerStar(self):
		"""
		Returns the total number of particles per star.
		"""

		armsPerStar = self.getArmCountPerStar()
		return 1 + armsPerStar * self.getTotalParticleCountPerArm()


	def getTotalParticleCount(self):
		"""
		Returns the total number of particles in all stars.
		"""

		if "getTotalParticleCount" in self._caches:
			return self._caches["getTotalParticleCount"]

		ret = self.getStarCount() * self.getTotalParticleCountPerStar()
		self._caches["getTotalParticleCount"] = ret
		return ret


	def getParticleType(self, index):
		"""
		Returns the type of particle for the given particle index.

		@throw IndexError
		       Throws if `index` is out of range.
		@throw TypeError
		       Throws if `index` is not an integer.

		@param[in] index
		           The particle index, as an integer in the range
		           [0, `getTotalParticleCount()` - 1].

		@return Returns "Core" for a core particle, "Arm" for an arm particle,
		        and "Magnetic" for a magnetic end particle.
		"""

		if not isinstance(index, int):
			raise TypeError()


		if index in self._caches["getParticleType"]:
			return self._caches["getParticleType"][index]


		if index < 0 or index >= self.getTotalParticleCount():
			raise IndexError()

		originalIndex = index
		index = index % self.getTotalParticleCountPerStar()

		if index == 0:
			self._caches["getParticleType"][originalIndex] = "Core"
			return "Core"

		index -= 1

		index = index % self.getTotalParticleCountPerArm()

		if index == self.getArmParticlesPerArm():
			self._caches["getParticleType"][originalIndex] = "Magnetic"
			return "Magnetic"

		self._caches["getParticleType"][originalIndex] = "Arm"
		return "Arm"


	def getParticleStructureIndices(self, particleID):
		"""
		Returns the type of particle for the given particle index.

		@warning The returned value is a reference to an internally cached
		         object. Do not modify!

		@throw IndexError
		       Throws if `particleID` is out of range.
		@throw TypeError
		       Throws if `particleID` is not an integer.

		@param[in] particleID
		           The particle index, as an integer in the range
		           [0, `getTotalParticleCount()` - 1].

		@return Returns a list of integers. The first integer corresponds to the
		        number of the star the given particle belongs to (ranging from
		        `0` to `getStarCount() - 1`). If the particle is a `Core`
		        particle, the following indices will be `None`; otherwise, the
		        next index corresponds to the arm the particle corresponds to
		        (in the range `0` to `getArmCountPerStar() - 1`). The last index
		        is `None` if the particle is a `Magnetic` particle, and
		        otherwise (i.e. if it is an `Arm` particle) corresponds to the
		        position in the arm, in the range form `0` (for particle closest
		        to the `Core`) to `getArmParticlesPerArm() - 1`.
		"""

		if not isinstance(particleID, int):
			raise TypeError()

		if particleID in self._caches["getParticleStructureIndices"]:
			return self._caches["getParticleStructureIndices"][particleID]

		if particleID < 0 or particleID >= self.getTotalParticleCount():
			raise IndexError()

		originalParticleID = particleID

		star = particleID // self.getTotalParticleCountPerStar()
		particleID = particleID % self.getTotalParticleCountPerStar()

		if particleID == 0:
			ret = [star, None, None]
			self._caches["getParticleStructureIndices"][originalParticleID] = \
				ret
			return ret

		particleID -= 1

		arm = particleID // self.getTotalParticleCountPerArm()
		particleID = particleID % self.getTotalParticleCountPerArm()

		if self.hasMagneticParticles():
			if particleID == self.getArmParticlesPerArm():
				return [star, arm, None]

		ret = [star, arm, particleID]
		self._caches["getParticleStructureIndices"][originalParticleID] = ret
		return ret


	def particlesAreBonded(self, pID1, pID2):
		"""
		Returns whether the two particles given are bonded.

		If `pID1 == pID2`, `False` is returned.

		@throw IndexError
		       Throws if `pID1` or `pID2` are out of range.
		@throw TypeError
		       Throws if `pID1` or `pID2` are not an integer.

		@param[in] pID1
		           The first particle index, as an integer in the range
		           [0, `getTotalParticleCount()` - 1].
		@param[in] pID2
		           The second particle index, as an integer in the range
		           [0, `getTotalParticleCount()` - 1].
		"""

		if not isinstance(pID1, int):
			raise TypeError()

		if not isinstance(pID2, int):
			raise TypeError()


		if (pID1, pID2) in self._caches["particlesAreBonded"]:
			return self._caches["particlesAreBonded"][(pID1, pID2)]


		if pID1 < 0 or pID1 >= self.getTotalParticleCount():
			raise IndexError()

		if pID2 < 0 or pID2 >= self.getTotalParticleCount():
			raise IndexError()


		if pID1 == pID2:
			self._caches["particlesAreBonded"][(pID1, pID2)] = False
			return False

		indices1 = self.getParticleStructureIndices(pID1)
		indices2 = self.getParticleStructureIndices(pID2)

		if indices1[0] != indices2[0]:
			self._caches["particlesAreBonded"][(pID1, pID2)] = False
			return False

		if indices1[1] is None or indices2[1] is None:
			if indices1[1] is None:
				nonCore = indices2
			else:
				nonCore = indices1

			if self.getArmParticlesPerArm() == 0 \
			   and self.hasMagneticParticles():
				self._caches["particlesAreBonded"][(pID1, pID2)] = True
				return True

			return nonCore[2] == 0

		if indices1[1] != indices2[1]:
			self._caches["particlesAreBonded"][(pID1, pID2)] = False
			return False

		if indices1[2] is None or indices2[2] is None:
			if indices1[2] is None:
				if indices2[2] is None:
					self._caches["particlesAreBonded"][(pID1, pID2)] = False
					return False
				nonMagnetic = indices2
			else:
				nonMagnetic = indices1

			ret = nonMagnetic[2] == self.getArmParticlesPerArm() - 1
			self._caches["particlesAreBonded"][(pID1, pID2)] = ret
			return ret


		ret = abs(indices1[2] - indices2[2]) == 1
		self._caches["particlesAreBonded"][(pID1, pID2)] = ret
		return ret


	def setParticles(self, particles):
		"""
		Sets the collection of particles to use for dynamic calculations.

		@throw TypeError
		       Throws if `particles` is not an instance of `ParticleCollection`.
		@throw ValueError
		       Throws if `particles` does not conatin the right number of
		       particles.

		@param[in] particles
		           An instance of `ParticleCollection` containing
		           `getTotalParticleCount` particles.
		           Only a reference to this object is saved!
		"""

		if not isinstance(particles, ParticleCollection):
			raise TypeError()

		if particles.getParticleCount() != self.getTotalParticleCount():
			raise ValueError()

		self._particles = particles


	def getMagneticClusters(self, magneticClusterMaxDistance):
		"""
		Returns the magnetic clusters, i.e. the largest groups of magnetic
		particles that have the property that from any one member of a magnetic
		cluster to any other member of that same cluster, there is a sequence of
		cluster members between the two magnetic particles such that the
		distance between consecutive members is at most
		`magneticClusterMaxDistance`.

		@throw TypeError
		       Throws if `magneticClusterMaxDistance` is neither `int` nor
		       `float`.
		@throw ValueError
		       Throws if no magnetic particles have been configured.
		@throw ValueError
		       Throws if `setParticles` has not been called previously.
		@throw ValueError
		       Throws if `magneticClusterMaxDistance` is negative.

		@param[in] magneticClusterMaxDistance
		           The maximum distance between magnetic particles that defines
		           a cluster. Must be a non-negative `int` or `float`.

		@return Returns a list of clusters, where each cluster is represented as
		        a list containing the positions (as instances of `Vector3DReal`)
		        of the cluster members.
		"""

		if not isinstance(magneticClusterMaxDistance, (int, float)):
			raise TypeError()

		if magneticClusterMaxDistance < 0:
			raise ValueError()

		if not self.hasMagneticParticles():
			raise ValueError()

		if self._particles is None:
			raise ValueError()

		magnetPositions = []
		for index in range(0, self._particles.getParticleCount()):
			if self.getParticleType(index) == "Magnetic":
				magnetPositions.append(self._particles.getPosition(index))

		clusters = [[pos] for pos in magnetPositions]

		return self._mergeClusters(clusters, magneticClusterMaxDistance)


	def getMagneticClusterCount(self, magneticClusterMaxDistance):
		"""
		Returns the number of magnetic clusters.

		See `getMagneticClusters` for further documentation.

		@throw TypeError
		       Throws if `magneticClusterMaxDistance` is neither `int` nor
		       `float`.
		@throw ValueError
		       Throws if no magnetic particles have been configured.
		@throw ValueError
		       Throws if `setParticles` has not been called previously.
		@throw ValueError
		       Throws if `magneticClusterMaxDistance` is negative.
		"""

		return len(self.getMagneticClusters(magneticClusterMaxDistance))


	def getWCAPotentialParameterEpsilon(self, type1, type2):
		r"""
		Returns the WCA potential parameter \f$ \epsilon \f$ for the interaction
		of particles of types `type1` and `type2`.

		@throw TypeError
		       Throws if either `type1` or `type2` are not of type `str`.
		@throw ValueError
		       Throws if either `type1` or `type2` have illegal values.

		@param[in] type1
		           The type of one of the particles, which must be one of
		           `"Core"`, `"Arm"`, or `"Magnetic"` (the latter being allowed
		           only if `hasMagneticParticles()`).
		@param[in] type2
		           The type of the other particle; see `type1` for further
		           information.
		"""

		for t in [type1, type2]:
			if not isinstance(t, str):
				raise TypeError
			allowedValues = ["Core", "Arm"]
			if self.hasMagneticParticles():
				allowedValues.append("Magnetic")
			if t not in allowedValues:
				raise ValueError()

		epsilon1 = \
			self._config["interactionParameters.epsilon_" + type1.lower()]
		epsilon2 = \
			self._config["interactionParameters.epsilon_" + type2.lower()]

		import math
		return math.sqrt(epsilon1 * epsilon2)


	def getWCAPotentialParameterSigma(self, type1, type2):
		r"""
		Returns the WCA potential parameter \f$ \sigma \f$ for the interaction
		of particles of types `type1` and `type2`.

		@throw TypeError
		       Throws if either `type1` or `type2` are not of type `str`.
		@throw ValueError
		       Throws if either `type1` or `type2` have illegal values.

		@param[in] type1
		           The type of one of the particles, which must be one of
		           `"Core"`, `"Arm"`, or `"Magnetic"` (the latter being allowed
		           only if `hasMagneticParticles()`).
		@param[in] type2
		           The type of the other particle; see `type1` for further
		           information.
		"""

		for t in [type1, type2]:
			if not isinstance(t, str):
				raise TypeError
			allowedValues = ["Core", "Arm"]
			if self.hasMagneticParticles():
				allowedValues.append("Magnetic")
			if t not in allowedValues:
				raise ValueError()

		sigma1 = self._config["interactionParameters.sigma_" + type1.lower()]
		sigma2 = self._config["interactionParameters.sigma_" + type2.lower()]

		return (sigma1 + sigma2) / 2.0


	def getWCAPotentialParameterD(self, type1, type2):
		r"""
		Returns the WCA potential parameter \f$ D \f$ for the interaction
		of particles of types `type1` and `type2`.

		@throw TypeError
		       Throws if either `type1` or `type2` are not of type `str`.
		@throw ValueError
		       Throws if either `type1` or `type2` have illegal values.

		@param[in] type1
		           The type of one of the particles, which must be one of
		           `"Core"`, `"Arm"`, or `"Magnetic"` (the latter being allowed
		           only if `hasMagneticParticles()`).
		@param[in] type2
		           The type of the other particle; see `type1` for further
		           information.
		"""

		for t in [type1, type2]:
			if not isinstance(t, str):
				raise TypeError
			allowedValues = ["Core", "Arm"]
			if self.hasMagneticParticles():
				allowedValues.append("Magnetic")
			if t not in allowedValues:
				raise ValueError()

		d1 = self._config["interactionParameters.D_" + type1.lower()]
		d2 = self._config["interactionParameters.D_" + type2.lower()]

		return (d1 + d2) / 2.0


	def getWCAPotential(self, type1, type2):
		"""
		Returns the WCA potential for the interaction of particles of types
		`type1` and `type2`.

		@warning The returned value is a reference to an internally cached
		         object. Do not modify!

		@throw TypeError
		       Throws if either `type1` or `type2` are not of type `str`.
		@throw ValueError
		       Throws if either `type1` or `type2` have illegal values.

		@param[in] type1
		           The type of one of the particles, which must be one of
		           `"Core"`, `"Arm"`, or `"Magnetic"` (the latter being allowed
		           only if `hasMagneticParticles()`).
		@param[in] type2
		           The type of the other particle; see `type1` for further
		           information.
		"""

		if (type1, type2) in self._caches["getWCAPotential"]:
			return self._caches["getWCAPotential"][(type1, type2)]

		epsilon = self.getWCAPotentialParameterEpsilon(type1, type2)
		sigma = self.getWCAPotentialParameterSigma(type1, type2)
		d = self.getWCAPotentialParameterD(type1, type2)

		from .PairPotentials.WeeksChandlerAndersen_DistanceOffset\
			import WeeksChandlerAndersen_DistanceOffset as WCA

		ret = WCA(epsilon, sigma, d)
		self._caches["getWCAPotential"][(type1, type2)] = ret
		return ret


	def getFENEPotential(self, type1, type2):
		"""
		Returns the WCA potential for the interaction of particles of types
		`type1` and `type2`.

		@warning The returned value is a reference to an internally cached
		         object. Do not modify!

		@throw TypeError
		       Throws if either `type1` or `type2` are not of type `str`.
		@throw ValueError
		       Throws if either `type1` or `type2` have illegal values.

		@param[in] type1
		           The type of one of the particles, which must be one of
		           `"Core"`, `"Arm"`, or `"Magnetic"` (the latter being allowed
		           only if `hasMagneticParticles()`).
		@param[in] type2
		           The type of the other particle; see `type1` for further
		           information.
		"""

		if (type1, type2) in self._caches["getFENEPotential"]:
			return self._caches["getFENEPotential"][(type1, type2)]

		epsilon = self.getWCAPotentialParameterEpsilon(type1, type2)
		sigma = self.getWCAPotentialParameterSigma(type1, type2)
		d = self.getWCAPotentialParameterD(type1, type2)

		from .PairPotentials.FENE import FENE

		K = 30 * epsilon / (sigma * sigma)
		l_0 = d
		R = 1.5 * sigma

		ret = FENE(K, l_0, R)
		self._caches["getFENEPotential"][(type1, type2)] = ret
		return ret


	def getMagneticPotential(self):
		"""
		Returns the magnetic dipole-dipole interaction potential.

		@warning The returned value is a reference to an internally cached
		         object. Do not modify!

		@throw RuntimeError
		       Throws if `not hasMagneticParticles()`.
		"""

		if not self.hasMagneticParticles():
			raise RuntimeError()

		if self._caches["getMagneticPotential"] is not None:
			return self._caches["getMagneticPotential"]

		from .PairPotentials.MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles \
			import MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles \
			as Potential
		from .Vector3DReal import Vector3DReal

		prefactor = self._config["interactionParameters.magneticPrefactor"]
		orientationX = self._config["interactionParameters.dipoleOrientation.[0]"]
		orientationY = self._config["interactionParameters.dipoleOrientation.[1]"]
		orientationZ = self._config["interactionParameters.dipoleOrientation.[2]"]
		orientation = Vector3DReal(orientationX, orientationY, orientationZ)

		ret = Potential(prefactor, orientation)
		self._caches["getMagneticPotential"] = ret
		return ret


	def getPotentialEnergy(self):
		"""
		Computes the potential energy of the current system.

		@throw ValueError
		       Throws if `setParticles` has not been called previously.
		"""

		if self._particles is None:
			raise ValueError()

		if self.getStarCount() != 1:
			raise NotImplementedError()


		ret = 0.0
		magnetic = self.getMagneticPotential()

		for pID1 in range(0, self.getTotalParticleCount()):
			type1 = self.getParticleType(pID1)
			pos1 = self._particles.getPosition(pID1)
			for pID2 in range(pID1 + 1, self.getTotalParticleCount()):
				type2 = self.getParticleType(pID2)
				pos2 = self._particles.getPosition(pID2)

				r = pos1 - pos2

				wca = self.getWCAPotential(type1, type2)
				ret += wca.getPotential(r)

				if self.particlesAreBonded(pID1, pID2):
					fene = self.getFENEPotential(type1, type2)
					ret += fene.getPotential(r)

				if type1 == "Magnetic" and type2 == "Magnetic":
					ret += magnetic.getPotential(r)

		return ret


	def _mergeClusters(self, clusters, magneticClusterMaxDistance):
		for cluster in clusters:
			for otherCluster in clusters:
				if otherCluster == cluster:
					continue

				for position in cluster:
					for otherPosition in otherCluster:
						delta = position - otherPosition
						distance = delta.getLength()
						if distance <= magneticClusterMaxDistance:
							newClusters = []
							for newCluster in clusters:
								if newCluster == cluster:
									continue
								if newCluster == otherCluster:
									continue
								newClusters.append(newCluster)
							newClusters.append(cluster + otherCluster)

							return \
								self._mergeClusters(
									newClusters, magneticClusterMaxDistance)

		return clusters
