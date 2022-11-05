# coding=utf-8

class EckartSystem:
	"""
	Represents an Eckart system.

	The Eckart system is defined as in the article
	"Eckart vectors, Eckart frames, and polyatomic molecules"
	by James D. Louck and Harold W. Galbraith,
	Reviews of Modern Physics, January 1976, Vol. 48, No. 1, pp. 69-106,
	DOI: 10.1103/RevModPhys.48.69
	"""

	def __init__(self, referenceConfiguration):
		"""
		The constructor.

		@throw TypeError
		       Throws if `referenceConfiguration` is not of type
		       `ParticleCollection`.
		@throw ValueError
		       Throws if `referenceConfiguration` does not have masse set for
		       all its particles.

		@param[in] referenceConfiguration
		           The configuration of particles to use as a constant reference
		           for the remainder of this instance's existence.
		"""

		from .ParticleCollection import ParticleCollection

		if not isinstance(referenceConfiguration, ParticleCollection):
			raise TypeError()

		for i in range(0, referenceConfiguration.getParticleCount()):
			try:
				referenceConfiguration.getMass(i)
			except RuntimeError:
				raise ValueError

		import copy
		self._referenceConfiguration = copy.deepcopy(referenceConfiguration)
		self._referenceConfiguration.shiftToCenterOfMassFrame()


	def getParticleCount(self):
		"""
		Returns the number of particles in the Eckart system.
		"""

		return self._referenceConfiguration.getParticleCount()


	def getReferencePosition(self, i):
		"""
		Returns the reference position vector \f$ \vec{a}_i \f$ for the particle
		with the index `i`.

		@throw KeyError
		       Throws if `i` is negative, or equal to or greater than the
		       number of `getParticleCount()`.
		@throw TypeError
		       Throws if `i` is not of type `int`.

		@param[in] i
		           The particle index to query, which must be an `int` in the
		           range `[0, getParticleCount() - 1]`.

		@return Returns an instance of `Vector3DReal`.
		"""

		if not isinstance(i, int):
			raise TypeError()

		if i < 0 or i >= self.getParticleCount():
			raise KeyError()

		return self._referenceConfiguration.getPosition(i)


	def getEckartVectors(self, instantaneousConfiguration):
		"""
		Returns the Eckart vectors \f$ \vec{F}_1, \vec{F}_2, \vec{F}_3 \f$
		defined by
		\f[
			\vec{F}_i = \sum_{\alpha = 1}^N m_\alpha a_i^\alpha \vec{r}^\alpha
		\f]
		where \f$ N \f$ is the number of particles, \f$ m_\alpha \f$ is the mass
		of the particle with index \f$ \alpha \f$, \f$ \vec{r}^\alpha \f$ is its
		instantaneous position as specified in `instantaneousConfiguration`, and
		\f$ a_i^\alpha \f$ is the \f$ i \f$-th component of the position vector
		of the \f$ \alpha \f$-th particle in the center-of-mass-frame of the
		reference configuration.

		@throw TypeError
		       Throws if `instantaneousConfiguration` is not an instance of
		       `ParticleCollection`.
		@throw ValueError
		       Throws if `instantaneousConfiguration` is incompatible with the
		       reference configuration, i.e. if the number of particles, or
		       their masses, mismatch.

		@param[in] instantaneousConfiguration
		           An instance of `ParticleCollection` containing the current
		           particle positions.

		@return Returns a list containing
		        \f$ \vec{F}_1, \vec{F}_2, \vec{F}_3 \f$, in that order, as
		        instances of `Vector3DReal`.
		"""

		from .ParticleCollection import ParticleCollection
		from .Vector3DReal import Vector3DReal

		if not isinstance(instantaneousConfiguration, ParticleCollection):
			raise TypeError()

		particleCount = instantaneousConfiguration.getParticleCount()
		if particleCount != self._referenceConfiguration.getParticleCount():
			raise ValueError()


		ret = [Vector3DReal(0, 0, 0) for _ in range(0, 3)]
		for i in range(0, particleCount):
			mass = instantaneousConfiguration.getMass(i)
			if mass != self._referenceConfiguration.getMass(i):
				raise ValueError()

			instantaneousPosition = instantaneousConfiguration.getPosition(i)
			referencePosition = self._referenceConfiguration.getPosition(i)
			for coord in range(0, 3):
				factor = mass * referencePosition[coord]
				ret[coord] += instantaneousPosition * factor


		return ret


	def getGramMatrix(
		self, instantaneousConfiguration,
		_eckartVectors = None):
		"""
		Returns the Gram matrix \f$ F \f$ with entries
		\f$ F_{ij} = \vec{F}_i \cdot \vec{F}_j \f$, \f$ \vec{F}_i \f$ being the
		vectors as returned by `getEckartVectors()`.

		@throw TypeError
		       Throws if `instantaneousConfiguration` is not an instance of
		       `ParticleCollection`.
		@throw ValueError
		       Throws if `instantaneousConfiguration` is incompatible with the
		       reference configuration, i.e. if the number of particles, or
		       their masses, mismatch.

		@param[in] instantaneousConfiguration
		           An instance of `ParticleCollection` containing the current
		           particle positions.
		@param[in] _eckartVectors
		           If not `None`, the given argument will be used as if returned
		           by `getEckartVectors` with the given
		           `instantaneousConfiguration`, and the latter's validity is
		           not checked.
		           This argument is meant to be used only from other functions
		           of this class.

		@return Returns a `3 x 3` matrix, in the form of a list of three lists
		        consisting of three `float` instances each.
		"""

		if _eckartVectors is None:
			eckartVectors = self.getEckartVectors(instantaneousConfiguration)
		else:
			eckartVectors = _eckartVectors

		ret = [[0, 0, 0] for _ in range(0, 3)]
		for i in range(0, 3):
			for j in range(i, 3):
				ret[i][j] = eckartVectors[i].dot(eckartVectors[j])

		ret[1][0] = ret[0][1]
		ret[2][0] = ret[0][2]
		ret[2][1] = ret[1][2]

		return ret


	def getEckartFrame(self, instantaneousConfiguration):
		"""
		Returns the Eckart frame vectors \f$ \hat{f}_1, \hat{f}_2, \hat{f}_3 \f$
		defined by
		\f[
			\left[ \hat{f}_1, \hat{f}_2, \hat{f}_3 \right]
			=
			\left[ \vec{F}_1, \vec{F}_2, \vec{F}_3 \right] F^{-1/2}
		\f]
		where the \f$ \vec{F}_i \f$ are the Eckart vectors as returned by
		`getEckartVectors`, and \f$ F \f$ is the Gram matrix as returned by
		`getGramMatrix`.

		@throw TypeError
		       Throws if `instantaneousConfiguration` is not an instance of
		       `ParticleCollection`.
		@throw ValueError
		       Throws if `instantaneousConfiguration` is incompatible with the
		       reference configuration, i.e. if the number of particles, or
		       their masses, mismatch.

		@param[in] instantaneousConfiguration
		           An instance of `ParticleCollection` containing the current
		           particle positions.

		@return Returns a list containing
		        \f$ \hat{f}_1, \hat{f}_2, \hat{f}_3 \f$, in that order, as
		        instances of `Vector3DReal`.
		"""

		ev = self.getEckartVectors(instantaneousConfiguration)

		left = []
		left.append([ev[0][0], ev[1][0], ev[2][0]])
		left.append([ev[0][1], ev[1][1], ev[2][1]])
		left.append([ev[0][2], ev[1][2], ev[2][2]])

		right = \
			self._getGramMatrixInverseSquareRoot(
				instantaneousConfiguration,
				_eckartVectors = ev)

		import numpy
		result = numpy.dot(left, right)

		from .Vector3DReal import Vector3DReal
		f1 = Vector3DReal(result[0][0], result[1][0], result[2][0])
		f2 = Vector3DReal(result[0][1], result[1][1], result[2][1])
		f3 = Vector3DReal(result[0][2], result[1][2], result[2][2])

		return [f1, f2, f3]


	def getEckartFrameEquilibriumPositions(self, instantaneousConfiguration):
		"""
		Returns the vectors \f$ \vec{c}^\alpha \f$,
		\f$ \alpha \in \left[0, N\right] \f$,
		in the laboratory frame coordinate system,
		where the \f$ \vec{c} \f$ are defined by
		\f[
			\vec{c}^\alpha = \sum_{i = 1}^3 a_i^\alpha \hat{f}_i
		\f]
		where \f$ N \f$ is the number of particles, \f$ a_i^\alpha \f$ is the
		\f$ i \f$-th component of the position vector of the \f$ \alpha \f$-th
		particle in the center-of-mass-frame of the reference configuration,
		and the \f$ \hat{f}_i \f$ are the Eckart frame vectors, as returned by
		`getEckartFrame(instantaneousConfiguration)`.

		@throw TypeError
		       Throws if `instantaneousConfiguration` is not an instance of
		       `ParticleCollection`.
		@throw ValueError
		       Throws if `instantaneousConfiguration` is incompatible with the
		       reference configuration, i.e. if the number of particles, or
		       their masses, mismatch.

		@param[in] instantaneousConfiguration
		           An instance of `ParticleCollection` containing the current
		           particle positions.

		@return Returns a list containing
		        \f$ \vec{c}^1, \vec{c}^2, \ldots, \vec{c}^N \f$, in that order,
		        as instances of `Vector3DReal`.
		"""

		result = []
		eckartFrame = self.getEckartFrame(instantaneousConfiguration)

		from .Vector3DReal import Vector3DReal
		for alpha in range(0, self.getParticleCount()):
			v = Vector3DReal(0, 0, 0)
			a = self.getReferencePosition(alpha)
			for i in range(0, 3):
				v += eckartFrame[i] * a[i]
			result.append(v)

		return result


	def getEckartMomentOfInertiaTensor(
		self, instantaneousConfiguration,
		_eckartFrameEquilibriumPositions = None):
		"""
		Returns the Eckart moment of inertia tensor \f$ J \f$ in the laboratory
		frame coordinate system, defined as
		\f[
			J =
			\sum_{\alpha = 1}^N
			\left(
				m_\alpha
				\left(
					\left( \vec{r}_\alpha - \vec{r}_{cm} \right)
					\cdot \vec{c}_\alpha
				\right)
				I
				-
				\left( \vec{r}_\alpha - \vec{r}_{cm} \right)
				\otimes \vec{c}_\alpha
			\right)
		\f]
		where \f$ N \f$ is the number of particles, \f$ I \f$ is the
		\f$ 3 \times 3 \f$ unit matrix, \f$ \otimes \f$ denotes the outer
		product, \f$ \vec{r}_\alpha \f$ is the instantaneous position of the
		\f$ \alpha \f$-th particle, \f$ \vec{r}_{cm} \f$ is the instantaneous
		position of the center of mass of the particles, \f$ \vec{c}_\alpha \f$
		are the Eckart frame equilibrium positions as returned by
		`getEckartFrameEquilibriumPositions(instantaneousConfiguration)`,
		and \f$ m_\alpha \f$ is the mass of the \f$ \alpha \f$-th particle.

		This quantity is defined as in equation (15) in the article
		"Application of the Eckart frame to soft matter: rotation of star
		polymers under shear flow"
		by Jurij Sablić, Rafael Delgado-Buscalioni, and Matej Praprotnik,
		arXiv:1707.09170v1 [cond-mat.soft]

		@throw TypeError
		       Throws if `instantaneousConfiguration` is not an instance of
		       `ParticleCollection`.
		@throw ValueError
		       Throws if `instantaneousConfiguration` is incompatible with the
		       reference configuration, i.e. if the number of particles, or
		       their masses, mismatch.

		@param[in] instantaneousConfiguration
		           An instance of `ParticleCollection` containing the current
		           particle positions.
		@param[in] _eckartFrameEquilibriumPositions
		           If not `None`, the given argument will be used as if returned
		           by `getEckartFrameEquilibriumPositions` with the given
		           `instantaneousConfiguration`, and the latter's validity is
		           not checked.
		           This argument is meant to be used only from other functions
		           of this class.

		@return Returns an instance of `numpy.ndarray` with shape `(3, 3)`.
		"""

		if _eckartFrameEquilibriumPositions is not None:
			cs = _eckartFrameEquilibriumPositions
		else:
			cs = \
				self.getEckartFrameEquilibriumPositions(
					instantaneousConfiguration)

		import numpy
		J = numpy.zeros((3, 3))

		centerOfMass = instantaneousConfiguration.getCenterOfMass()
		unitMatrix = numpy.identity(3)
		for alpha in range(0, self.getParticleCount()):
			m = instantaneousConfiguration.getMass(alpha)
			R = instantaneousConfiguration.getPosition(alpha) - centerOfMass
			c = cs[alpha]

			outerProduct = numpy.zeros((3, 3))
			for i in range(0, 3):
				for j in range(0, 3):
					outerProduct[i][j] = R[i] * c[j]

			current = numpy.dot(unitMatrix, R.dot(c))
			current -= outerProduct
			J += numpy.dot(current, m)

		return J


	def getEckartAngularVelocityVector(self, instantaneousConfiguration):
		"""
		Returns the Eckart angular velocity vector \f$ \vec{\Omega} \f$ in the
		laboratory frame coordinate system, defined as
		\f[
			\vec{\Omega} =
			J^{-1}
			\sum_{\alpha = 1}^N
			m_\alpha
			\vec{c}_\alpha
			\times
			\left( \vec{v}_\alpha - \vec{v}_{cm} \right)
		\f]
		where \f$ N \f$ is the number of particles, \f$ J \f$ is the
		Eckart moment of inertia tensor, as returned by
		`getEckartMomentOfInertiaTensor(instantaneousConfiguration)`,
		\f$ \times \f$ denotes the cross
		product, \f$ \vec{v}_\alpha \f$ is the instantaneous velocity of the
		\f$ \alpha \f$-th particle, \f$ \vec{v}_{cm} \f$ is the instantaneous
		velocity of the center of mass of the particles, \f$ \vec{c}_\alpha \f$
		are the Eckart frame equilibrium positions as returned by
		`getEckartFrameEquilibriumPositions(instantaneousConfiguration)`,
		and \f$ m_\alpha \f$ is the mass of the \f$ \alpha \f$-th particle.

		This quantity is defined as in equation (14) in the article
		"Application of the Eckart frame to soft matter: rotation of star
		polymers under shear flow"
		by Jurij Sablić, Rafael Delgado-Buscalioni, and Matej Praprotnik,
		arXiv:1707.09170v1 [cond-mat.soft]

		@throw TypeError
		       Throws if `instantaneousConfiguration` is not an instance of
		       `ParticleCollection`.
		@throw ValueError
		       Throws if `instantaneousConfiguration` is incompatible with the
		       reference configuration, i.e. if the number of particles, or
		       their masses, mismatch.

		@param[in] instantaneousConfiguration
		           An instance of `ParticleCollection` containing the current
		           particle positions.

		@return Returns an instance of `Vector3DReal`.
		"""

		cs = self.getEckartFrameEquilibriumPositions(instantaneousConfiguration)
		J = \
			self.getEckartMomentOfInertiaTensor(
				instantaneousConfiguration,
				_eckartFrameEquilibriumPositions = cs)
		comVelocity = instantaneousConfiguration.getCenterOfMassVelocity()

		from .Vector3DReal import Vector3DReal
		rhs = Vector3DReal(0, 0, 0)

		for alpha in range(0, self.getParticleCount()):
			relativeVelocity = \
				instantaneousConfiguration.getVelocity(alpha) - comVelocity

			current = cs[alpha].cross(relativeVelocity)
			rhs += current * instantaneousConfiguration.getMass(alpha)

		rhs = [rhs[i] for i in range(0, 3)]

		import numpy
		import scipy.linalg
		result = numpy.dot(scipy.linalg.inv(J, overwrite_a = True), rhs)

		return Vector3DReal(result[0], result[1], result[2])


	def _getGramMatrixInverseSquareRoot(
		self, instantaneousConfiguration,
		_eckartVectors = None):
		"""
		Returns \f$ F^{-1/2} \f$, where \f$ F \f$ is the Gram matrix as returned
		by `getGramMatrix()`.

		@throw TypeError
		       Throws if `instantaneousConfiguration` is not an instance of
		       `ParticleCollection`.
		@throw ValueError
		       Throws if `instantaneousConfiguration` is incompatible with the
		       reference configuration, i.e. if the number of particles, or
		       their masses, mismatch.

		@param[in] instantaneousConfiguration
		           An instance of `ParticleCollection` containing the current
		           particle positions.
		@param[in] _eckartVectors
		           If not `None`, the given argument will be used as if returned
		           by `getEckartVectors` with the given
		           `instantaneousConfiguration`, and the latter's validity is
		           not checked.
		           This argument is meant to be used only from other functions
		           of this class.

		@return Returns an instance of `numpy.ndarray` with shape `(3, 3)`.
		"""

		gramMatrix = \
			self.getGramMatrix(
				instantaneousConfiguration,
				_eckartVectors = _eckartVectors)

		import scipy.linalg
		invertedGramMatrix = scipy.linalg.inv(gramMatrix, overwrite_a = True)
		return scipy.linalg.sqrtm(invertedGramMatrix)



