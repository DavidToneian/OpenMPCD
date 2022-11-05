from .Vector3DReal import Vector3DReal

import math
import numpy
import numpy.linalg

class ParticleCollection:
	"""
	Represents the state of a collection of particles.
	"""

	def __init__(self):
		"""
		Constructs an empty particle collection.
		"""

		self.positions = []
		self.velocities = []
		self.uniformMass = None

		self.gyrationTensor = None


	def setPositionsAndVelocities(self, positions, velocities):
		"""
		Sets the particle positions and velocities.

		The given arguments will be copied into an instance-internal store.

		@throw TypeError
		       Throws if the arguments do not have the types specified in their
		       documentation.
		@throw ValueError
		       Throws if the `positions` and `velocities` lists do not have
		       equal length.

		@param[in] positions
		           A list of `Vector3DReal` instances, describing the particle
		           positions.
		@param[in] velocities
		           A list of `Vector3DReal` instances, describing the particle
		           velocities.
		"""

		if not isinstance(positions, list):
			raise TypeError()
		if not isinstance(velocities, list):
			raise TypeError()


		for x in positions:
			if not isinstance(x, Vector3DReal):
				raise TypeError()
		for x in velocities:
			if not isinstance(x, Vector3DReal):
				raise TypeError()


		if len(velocities) != 0:
			if len(positions) != len(velocities):
				raise ValueError("Arrays of different length given")


		import copy
		self.positions = copy.deepcopy(positions)
		self.velocities = copy.deepcopy(velocities)

		self.gyrationTensor = None


	def setUniformMass(self, mass):
		"""
		Sets a uniform mass for all particles, including currently stored ones
		and ones that may be added or altered later.

		@throw TypeError
		       Throws if `mass` is neither `int` nor `float`.
		@throw ValueError
		       Throws if `mass` is smaller than `0`.

		@param[in] mass
		           The particle mass, which must be a non-negative `int` or
		           `float`.
		"""

		if not isinstance(mass, (int, float)):
			raise TypeError()

		if mass < 0:
			raise ValueError()

		self.uniformMass = float(mass)


	def isEmpty(self):
		"""
		Returns whether this collection contains any particles.
		"""

		return len(self.positions) == 0


	def getParticleCount(self):
		"""
		Returns the number of particles in this collection.
		"""

		return len(self.positions)


	def getPositions(self):
		"""
		Returns a reference to the internal list of position vectors.

		@warning
		It is assumed that the returned reference will not be manipulated!
		"""

		return self.positions


	def getPosition(self, index):
		"""
		Returns the position of the particle with the given `index`, as an
		instance of `Vector3DReal`.

		@throw IndexError
		       Throws if `index` is negative or `index >= getParticleCount()`.
		@throw TypeError
		       Throws if `index` is not an `int`.

		@param[in] index
		           The index of the particle to query, in the range of
		           `[0, getParticleCount() - 1]`.
		"""

		if not isinstance(index, int):
			raise TypeError()

		if index < 0 or index >= self.getParticleCount():
			raise IndexError()

		return self.positions[index]


	def getVelocities(self):
		"""
		Returns a reference to the internal list of velocity vectors.

		@warning
		It is assumed that the returned reference will not be manipulated!
		"""

		return self.velocities


	def getVelocity(self, index):
		"""
		Returns the velocity of the particle with the given `index`, as an
		instance of `Vector3DReal`

		@throw IndexError
		       Throws if `index` is negative or `index >= getParticleCount()`.
		@throw TypeError
		       Throws if `index` is not an `int`.

		@param[in] index
		           The index of the particle to query, in the range of
		           `[0, getParticleCount() - 1]`.
		"""

		if not isinstance(index, int):
			raise TypeError()

		if index < 0 or index >= self.getParticleCount():
			raise IndexError()

		return self.velocities[index]


	def getMass(self, index):
		"""
		Returns the mass of the particle with the given `index` as a `float`.

		@throw IndexError
		       Throws if `index` is negative or `index >= getParticleCount()`.
		@throw RuntimeError
		       Throws if no mass has been specified for the given particle.
		@throw TypeError
		       Throws if `index` is not an `int`.

		@param[in] index
		           The index of the particle to query, in the range of
		           `[0, getParticleCount() - 1]`.
		"""

		if not isinstance(index, int):
			raise TypeError()

		if index < 0 or index >= self.getParticleCount():
			raise IndexError()

		if self.uniformMass is None:
			raise RuntimeError()

		return self.uniformMass


	def getMomentum(self, index):
		"""
		Returns the momentum of the particle with the given `index`, as an
		instance of `Vector3DReal`

		@throw IndexError
		       Throws if `index` is negative or `index >= getParticleCount()`.
		@throw RuntimeError
		       Throws if no mass has been specified for the given particle.
		@throw TypeError
		       Throws if `index` is not an `int`.

		@param[in] index
		           The index of the particle to query, in the range of
		           `[0, getParticleCount() - 1]`.
		"""

		return self.velocities[index] * self.getMass(index)


	def getCenterOfMass(self):
		"""
		Returns the center of mass of the particles, as an instance of
		`Vector3DReal`.

		@throw RuntimeError
		       Throws if `isEmpty()`.
		@throw RuntimeError
		       Throws if the mass has not been specified for all particles.
		"""

		if self.isEmpty():
			raise RuntimeError()

		cumulative = Vector3DReal(0, 0, 0)
		mass = 0.0

		for index, position in enumerate(self.positions):
			currentMass = self.getMass(index)
			mass += currentMass
			cumulative += position * currentMass

		centerOfMass = cumulative / mass
		return centerOfMass


	def getCenterOfPositions(self):
		"""
		Returns the unweighted average of all particle positions, as an instance
		of `Vector3DReal`.

		@throw RuntimeError
		       Throws if `isEmpty()`.
		"""

		if self.isEmpty():
			raise RuntimeError()

		cumulative = Vector3DReal(0, 0, 0)

		for position in self.positions:
			cumulative += position

		return cumulative / len(self.positions)


	def getCenterOfMassVelocity(self):
		"""
		Returns the velocity of the center of mass, as an instance of
		`Vector3DReal`.

		@throw RuntimeError
		       Throws if `isEmpty()`.
		@throw RuntimeError
		       Throws if the mass has not been specified for all particles.
		"""

		if self.isEmpty():
			raise RuntimeError()

		cumulative = Vector3DReal(0, 0, 0)
		mass = 0

		for index, velocity in enumerate(self.velocities):
			currentMass = self.getMass(index)
			mass += currentMass
			cumulative += velocity * currentMass

		centerOfMassVelocity = cumulative / mass
		return centerOfMassVelocity


	def rotateAroundNormalizedAxis(self, axis, angle):
		"""
		Rotates all position and velocity vectors around the given `axis` for
		the given `angle`.

		@throw TypeError
		       Throws if one of the arguments is of the wrong type.
		@throw ValueError
		       Throws if `axis` is not a unit-length vector.

		@param[in] axis
		           The axis to rotate around, which must have unit length.
		@param[in] angle
		           The angle to rotate, in radians, as an instance of `float`.
		"""

		if not isinstance(axis, Vector3DReal):
			raise TypeError()

		if not isinstance(angle, float):
			raise TypeError()

		if not axis.getLengthSquared() == 1.0:
			raise ValueError()


		for position in self.positions:
			position.rotateAroundNormalizedAxis(axis, angle)

		for velocity in self.velocities:
			velocity.rotateAroundNormalizedAxis(axis, angle)

		self.gyrationTensor = None


	def shiftToCenterOfMassFrame(self):
		"""
		Transforms the positions and velocities into the center of mass frame.
		"""

		centerOfMass = self.getCenterOfMass()
		centerOfMassVelocity = self.getCenterOfMassVelocity()

		for index, _ in enumerate(self.positions):
			self.positions[index] -= centerOfMass

		for index, _ in enumerate(self.velocities):
			self.velocities[index] -= centerOfMassVelocity

		self.gyrationTensor = None


	def getGyrationTensor(self):
		"""
		Returns the gyration tensor \f$ S \f$.

		The returned object is a \f$ 3 \times 3 \f$ symmetric matrix of type
		`numpy.ndarray`.
		Given \f$ N \f$ particles with positions \f$ \vec{r}^{(i)} \f$ (in any
		Cartesian coordinate frame), the \f$ \left( m, n \right) \f$-component
		of the gyration tensor is defined by
		\f[
			S_{mn}
			=
			\frac{ 1 }{ 2 N^2 }
			\sum_{i = 1}^N
			\sum_{j = 1}^N
			\left( \vec{r}^{(i)}_m - \vec{r}^{(j)}_m \right)
			\left( \vec{r}^{(i)}_n - \vec{r}^{(j)}_n \right)
		\f]

		@throw ValueError
		       Throws if there are no particles in this instance.
		"""

		if self.gyrationTensor is not None:
			return self.gyrationTensor

		if self.isEmpty():
			raise ValueError()


		S = numpy.zeros((3, 3))

		for r1 in self.positions:
			for r2 in self.positions:
				for m in range(0, 3):
					for n in range(m, 3):
						S[m][n] += (r1[m] - r2[m]) * (r1[n] - r2[n])

		factor = 1.0 / (2 * self.getParticleCount() * self.getParticleCount())
		for m in range(0, 3):
			for n in range(m, 3):
				S[m][n] *= factor

		S[1][0] = S[0][1]
		S[2][0] = S[0][2]
		S[2][1] = S[1][2]

		self.gyrationTensor = S
		return S


	def getMomentOfInertiaTensor(self):
		centerOfMass = self.getCenterOfMass()
		if centerOfMass.getLengthSquared() > 1e-20:
			raise ValueError(
				"The center of mass has to be at the origin! " +
				"Instead, it squared distance to the origin is: " +
				str(centerOfMass.getLengthSquared()))

		Ixx = 0.0
		Ixy = 0.0
		Ixz = 0.0
		Iyy = 0.0
		Iyz = 0.0
		Izz = 0.0

		for index, position in enumerate(self.positions):
			x = position.getX()
			y = position.getY()
			z = position.getZ()
			mass = self.getMass(index)

			Ixx += mass * (y * y + z * z)
			Ixy += -mass * x * y
			Ixz += -mass * x * z
			Iyy += mass * (x * x + z * z)
			Iyz += -mass * y * z
			Izz += mass * (x * x + y * y)

		I = numpy.array([
			[Ixx, Ixy, Ixz],
			[Ixy, Iyy, Iyz],
			[Ixz, Iyz, Izz]])

		return I

	def getTotalAngularMomentumVector(self):
		L = Vector3DReal(0, 0, 0)

		for index, position in enumerate(self.positions):
			L += position.cross(self.getMomentum(index))

		return L

	def getRotationFrequencyVector(self):
		I = self.getMomentOfInertiaTensor()
		L = self.getTotalAngularMomentumVector()
		L = numpy.array([L.getX(), L.getY(), L.getZ()])
		omega = numpy.linalg.inv(I).dot(L)
		return Vector3DReal(omega[0], omega[1], omega[2])


	def getGyrationTensorPrincipalMoments(self):
		"""
		Returns the eigenvalues \f$ \lambda_x^2 \f$, \f$ \lambda_y^2 \f$, and
		\f$ \lambda_z^2 \f$ of the gyration tensor (as returned by
		`getGyrationTensor`). The eigenvalues are arranged such that
		\f$ \lambda_x^2 \le \lambda_y^2 \le \lambda_y^z \f$.

		The eigenvalues are real and of type `numpy.float64`, and returned as
		the elements of a list.
		"""

		S = self.getGyrationTensor()
		eigenvalues, _ = numpy.linalg.eig(S)

		x, y, z = numpy.sort(eigenvalues)

		if not (numpy.isreal(x) and numpy.isreal(y) and numpy.isreal(z)):
			raise ValueError("Unexpected")

		return [x, y, z]


	def getGyrationTensorEigensystem(self):
		"""
		Returns the eigenvalues and eigenvectors of the gyration tensor (as
		returned by `getGyrationTensor`).

		The returned object is a list, with each element being a list of first
		the eigenvalue, and second the associated eigenvector. The tuples are
		sorted by the value of the eigenvalue, smallest first.

		The eigenvalues are real and of type `numpy.float64`, and the
		eigenvectors are of type `numpy.ndarray` with three entries of type
		`numpy.float64`.
		"""

		S = self.getGyrationTensor()
		eigenvalues, eigenvectors = numpy.linalg.eig(S)

		for eigenvalue in eigenvalues:
			if not numpy.isreal(eigenvalue):
				raise ValueError("Unexpected")

		indices = eigenvalues.argsort()

		ret = []
		for index in indices:
			ret.append([eigenvalues[index], eigenvectors[:, index]])

		return ret


	def getRadiusOfGyrationSquared(self):
		"""
		Returns the sum of the eigenvalues of the gyration tensor, as returned
		by `getGyrationTensorPrincipalMoments`
		"""

		x, y, z = self.getGyrationTensorPrincipalMoments()

		return x + y + z

	def getRadiusOfGyration(self):
		"""
		Returns the radius of gyration \f$ R_g \f$, i.e. the square root of the
		result of `getRadiusOfGyrationSquared`.
		"""

		return math.sqrt(self.getRadiusOfGyrationSquared())

	def getAsphericity(self):
		"""
		Returns the asphericity \f$ b \f$, i.e.
		\f[
			\lambda_z^2 - \frac{1}{2} \left( \lambda_x^2 + \lambda_y^2 \right)
		\f]
		where \f$ \lambda_z^2 \f$ is the largest eigenvalue of the gyration
		tensor, and \f$ \lambda_x^2 \f$ and \f$ \lambda_y^2 \f$ are the two
		smallest eigenvalues.
		"""

		x, y, z = self.getGyrationTensorPrincipalMoments()

		return z - 0.5 * (x + y)

	def getAcylindricity(self):
		"""
		Returns the acylindricity \f$ c \f$, i.e.
		\f[
			\lambda_y^2 - \lambda_x^2
		\f]
		where \f$ \lambda_x^2 \f$ is the smallest eigenvalue of the gyration
		tensor, \f$ \lambda_z^2 \f$ is the largest eigenvalue, and
		\f$ \lambda_y^2 \f$ is the one in between.
		"""

		x, y, _ = self.getGyrationTensorPrincipalMoments()

		return y - x

	def getRelativeShapeAnisotropy(self):
		"""
		Returns the relative shape anisotropy \f$ \kappa^2 \f$, i.e.
		\f[
			\frac{ b^2 + \frac{3}{4} c^2 }{ R_g^4 }
		\f]
		where \f$ b \f$ is the asphericity, \f$ c \f$ is the acylindricity, and
		\f$ R_g \f$ is the radius of gyration.
		"""

		b = self.getAsphericity()
		c = self.getAcylindricity()
		R_g_squared = self.getRadiusOfGyrationSquared()

		return (b * b + 0.75 * c * c) / (R_g_squared * R_g_squared)

	def getOrientationAngle(self, axis):
		"""
		Returns the angle between the given `axis` and the eigenvector of the
		gyration tensor that corresponds to the largest eigenvalue.
		"""

		if len(axis) != 3:
			raise ValueError("")

		eigenpairs = self.getGyrationTensorEigensystem()

		return self._getAngleBetweenLines(eigenpairs[2][1], axis)

	def getOrientationAngles(self, axis):
		"""
		Returns the angle between the given `axis` and the eigenvectors of the
		gyration tensor. The returned value is a list, sorted by increasing
		eigenvalues, of pairs of eigenvalue and corresponding angle.
		"""

		if len(axis) != 3:
			raise ValueError("")

		eigenpairs = self.getGyrationTensorEigensystem()

		ret = []
		for eigenpair in eigenpairs:
			eigenvalue = eigenpair[0]
			eigenvector = eigenpair[1]

			angle = self._getAngleBetweenLines(eigenvector, axis)

			ret.append([eigenvalue, angle])

		return ret



	def _getAngleBetweenLines(self, v1, v2):
		"""
		Returns the angle between two lines, characterized by the given two
		vectors.
		"""

		dot = numpy.dot(v1, v2)
		if dot < 0:
			dot *= -1

		len1Squared = numpy.dot(v1, v1)
		len2Squared = numpy.dot(v2, v2)

		return numpy.arccos(dot / numpy.sqrt(len1Squared * len2Squared))


	def __eq__(self, rhs):
		"""
		The equality operator.

		Returns whether this instance stores the same particles (i.e. their
		count, positions, velocities, and masses) as the given `rhs` instance.

		Having set a different uniform mass in two instances makes them not
		equal, even if there are no particles.

		@param[in] rhs The right-hand-side instance to compare to.

		@return Returns whether the two instances are equal, or `NotImplemented`
		        if the two instances are not of the same type.
		"""

		if not isinstance(rhs, self.__class__):
			return NotImplemented

		if len(self.positions) != len(rhs.positions):
			return False

		if self.uniformMass != rhs.uniformMass:
			return False

		for index, position in enumerate(self.positions):
			if position != rhs.positions[index]:
				return False
			if self.velocities[index] != rhs.velocities[index]:
				return False

		return True


	def __ne__(self, rhs):
		"""
		The inequality operator.

		@param[in] rhs The right-hand-side instance to compare to.

		@return Returns `NotImplemented` if the two instances are not of the
		        same type, and `not self == rhs` otherwise.
		"""

		if not isinstance(rhs, self.__class__):
			return NotImplemented

		return not self == rhs
