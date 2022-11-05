import math

class Vector3DReal:
	"""
	Represents a three-dimensional vector with the entries being real numbers.
	"""

	def __init__(self, x, y = None, z = None):
		"""
		The constructor.

		@throw TypeError
		       Throws if the types of the arguments do not match the
		       specification.

		@param[in] x
		           This parameter may be an instance of `Vector3DReal`, in which
		           case this instance will be constructed to contain the same
		           elements as `x`, and it will be required that `y` and `z` be
		           `None`.
		           Alternatively, `x` may be a list or tuple of length `3`, with
		           each element being either an `int` or a `float`; the element
		           `0` will be taken as the `x` value for this vector, the
		           element `1` as `y`, and the element `2` as `z`. In this case,
		           too, `y` and `z` must be `None`.
		           Alternatively, `x` may be an `int` or `float` and will serve
		           as the `x` value for this instance. In this case, `y` and `z`
		           must be set likewise.
		@param[in] y
		           Either `None` or the `y` value for this vector; see the
		           documentation of `x`.
		@param[in] z
		           Either `None` or the `z` value for this vector; see the
		           documentation of `x`.
		"""

		if isinstance(x, Vector3DReal):
			if y is not None or z is not None:
				raise TypeError()

			self.x = x.x
			self.y = x.y
			self.z = x.z
			return

		allowedTypes = (int, float)

		if isinstance(x, (list, tuple)):
			if y is not None or z is not None:
				raise TypeError()

			if len(x) != 3:
				raise TypeError()

			for i in [0, 1, 2]:
				if not isinstance(x[i], allowedTypes):
					raise TypeError()

			self.x = x[0]
			self.y = x[1]
			self.z = x[2]
			return


		for var in [x, y, z]:
			if not isinstance(var, allowedTypes):
				raise TypeError()

		self.x = x
		self.y = y
		self.z = z


	def getX(self):
		"""
		Returns the `x` value.
		"""

		return self.x


	def getY(self):
		"""
		Returns the `y` value.
		"""

		return self.y


	def getZ(self):
		"""
		Returns the `z` value.
		"""

		return self.z


	def dot(self, rhs):
		"""
		Returns the dot product of this vector with `rhs`.

		@param[in] rhs
		           The right-hand-side instance, which must be of type
		           `Vector3DReal`.
		"""

		self._assertIsSameType(rhs)

		return self.x * rhs.x + self.y * rhs.y + self.z * rhs.z


	def cross(self, rhs):
		"""
		Returns the cross product of this vector with `rhs`.

		@param[in] rhs
		           The right-hand-side instance, which must be of type
		           `Vector3DReal`.
		"""

		self._assertIsSameType(rhs)

		cx = self.y * rhs.z - self.z * rhs.y;
		cy = self.z * rhs.x - self.x * rhs.z;
		cz = self.x * rhs.y - self.y * rhs.x;
		return Vector3DReal(cx, cy, cz)


	def getLengthSquared(self):
		"""
		Returns the square of the length of this vector.
		"""

		return self.dot(self)


	def getLength(self):
		"""
		Returns the length of this vector.
		"""

		return math.sqrt(self.getLengthSquared())


	def getNormalized(self):
		"""
		Returns this vector, but normalized, i.e. divided by its length.

		@throw ValueError
		       Throws if `getLengthSquared() == 0`.
		"""

		if self.getLengthSquared() == 0:
			raise ValueError()

		factor = 1.0 / self.getLength()
		return Vector3DReal(self.x * factor, self.y * factor, self.z * factor)

	def normalize(self):
		"""
		Normalizes this vector, i.e. divides it by its length.

		@throw ValueError
		       Throws if `getLengthSquared() == 0`.
		"""

		if self.getLengthSquared() == 0:
			raise ValueError()

		factor = 1.0 / self.getLength()
		self.x = self.x * factor
		self.y = self.y * factor
		self.z = self.z * factor


	def getProjectionOnto(self, rhs):
		"""
		Returns the projection of this vector onto the direction `rhs`.

		@throw ValueError
		       Throws if `rhs.getLengthSquared() == 0`.

		@param[in] rhs
		           The right-hand-side instance, which must be of type
		           `Vector3DReal` and must not be of zero length.
		"""

		self._assertIsSameType(rhs)

		if rhs.getLengthSquared() == 0:
			raise ValueError()

		normalized = rhs.getNormalized()
		return normalized * normalized.dot(self)


	def getPerpendicularTo(self, rhs):
		"""
		Returns the component of this vector that is perpendicular to `rhs`.

		@throw ValueError
		       Throws if `rhs.getLengthSquared() == 0`.

		@param[in] rhs
		           The right-hand-side instance, which must be of type
		           `Vector3DReal` and must not be of zero length.
		"""

		self._assertIsSameType(rhs)

		if rhs.getLengthSquared() == 0:
			raise ValueError()

		return self -self.getProjectionOnto(rhs)


	def getRotatedAroundNormalizedAxis(self, axis, angle):
		"""
		Returns the this vector, rotated around the normalized vector `axis` by
		`angle` radians in counter-clockwise direction.

		@throw ValueError
		       Throws if `axis.getLengthSquared() == 0`.

		@param[in] axis
		           The vector to rotate about, which must be of type
		           `Vector3DReal` and is assumed to be of unit length.
		@param[in] angle
		           The amount to rotate, in radians.
		"""

		self._assertIsSameType(axis)

		if axis.getLengthSquared() == 0:
			raise ValueError()

		thisDotAxis = self.dot(axis)
		axisCrossThis = axis.cross(self)
		projectionOntoAxis = axis * thisDotAxis
		return \
			projectionOntoAxis + \
			(self -projectionOntoAxis) * math.cos(angle) + \
			axisCrossThis * math.sin(angle)


	def rotateAroundNormalizedAxis(self, axis, angle):
		"""
		Sets this vector to be the result of `getRotatedAroundNormalizedAxis`,
		called with the given arguments.

		@throw ValueError
		       Throws if `axis.getLengthSquared() == 0`.

		@param[in] axis
		           The vector to rotate about, which must be of type
		           `Vector3DReal` and is assumed to be of unit length.
		@param[in] angle
		           The amount to rotate, in radians.
		"""

		self._assertIsSameType(axis)

		rotated = self.getRotatedAroundNormalizedAxis(axis, angle)
		self.x = rotated.x
		self.y = rotated.y
		self.z = rotated.z


	def isClose(self, rhs, relativeTolerance = None, absoluteTolerance = None):
		"""
		Returns whether this instance and `rhs` are close to each other.

		Two vectors are "close" in this sense if all of their components are
		close in the sense of `numpy.allclose`.

		@throw TypeError
		       Throws if any of the arguments have an invalid type.
		@throw ValueError
		       Throws if any of the arguments have an invalid value.

		@param[in] rhs
		           The other instance of `Vector3DReal` to compare to.
		@param[in] relativeTolerance
		           A non-negative `float` that is passed as the `rtol` parameter
		           of `numpy.allclose`, or `None` for its default value.
		@param[in] absoluteTolerance
		           A non-negative `float` that is passed as the `atol` parameter
		           of `numpy.allclose`, or `None` for its default value.
		"""

		self._assertIsSameType(rhs)

		for x in [relativeTolerance, absoluteTolerance]:
			if not isinstance(x, (float, type(None))):
				raise TypeError()
			if x is not None and x < 0:
				raise ValueError


		a = [self.x, self.y, self.z]
		b = [rhs.x, rhs.y, rhs.z]
		kwargs = {}
		if relativeTolerance is not None:
			kwargs["rtol"] = relativeTolerance
		if absoluteTolerance is not None:
			kwargs["atol"] = absoluteTolerance

		import numpy
		return numpy.allclose(a, b, **kwargs)


	def __eq__(self, rhs):
		"""
		Returns `True` if `rhs` contains the same entries as this instance, and
		`False` otherwise.

		@param[in] rhs
		           The right-hand-side instance, which must be of type
		           `Vector3DReal`.
		"""

		self._assertIsSameType(rhs)

		if self.x != rhs.x:
			return False
		if self.y != rhs.y:
			return False
		if self.z != rhs.z:
			return False
		return True


	def __ne__(self, rhs):
		"""
		Returns the negation of `__eq__`.

		@param[in] rhs
		           The right-hand-side instance, which must be of type
		           `Vector3DReal`.
		"""

		return not self.__eq__(rhs)


	def __add__(self, rhs):
		"""
		Returns the sum of this instance and `rhs`.

		@param[in] rhs
		           The right-hand-side instance, which must be of type
		           `Vector3DReal`.
		"""

		self._assertIsSameType(rhs)

		return Vector3DReal(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)


	def __sub__(self, rhs):
		"""
		Returns the difference of this instance and `rhs`.

		@param[in] rhs
		           The right-hand-side instance, which must be of type
		           `Vector3DReal`.
		"""

		self._assertIsSameType(rhs)

		return Vector3DReal(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)


	def __mul__(self, rhs):
		"""
		Returns the product of this instance and `rhs`.

		@param[in] rhs
		           The factor to multiply with, which must be either of type
		           `int` or `float`.
		"""

		allowedTypes = (int, float)
		if not isinstance(rhs, allowedTypes):
			raise TypeError()

		return Vector3DReal(self.x * rhs, self.y * rhs, self.z * rhs)


	def __div__(self, rhs):
		"""
		Returns the ratio of this instance and `rhs`.

		The division performed is always a floating-point division.

		@param[in] rhs
		           The factor to divide by with, which must be either of type
		           `int` or `float`, and must not be `0`.
		"""

		return self.__truediv__(rhs)


	def __truediv__(self, rhs):
		"""
		Returns the ratio of this instance and `rhs`.

		The division performed is always a floating-point division.

		@param[in] rhs
		           The factor to divide by with, which must be either of type
		           `int` or `float`, and must not be `0`.
		"""

		allowedTypes = (int, float)
		if not isinstance(rhs, allowedTypes):
			raise TypeError()

		if rhs == 0:
			raise ValueError()

		if isinstance(rhs, int):
			rhs = float(rhs)

		return Vector3DReal(self.x / rhs, self.y / rhs, self.z / rhs)


	def __neg__(self):
		"""
		Returns the negative of this vector.
		"""

		return Vector3DReal(-self.x, -self.y, -self.z)


	def __getitem__(self, index):
		"""
		Returns the coordinate with the given index.

		@param[in] index
		           The coordinate index, which must be `0` for the `x`
		           coordinate, `1` for `y`, or `2` for `z`.
		"""

		if index == 0:
			return self.getX()

		if index == 1:
			return self.getY()

		if index == 2:
			return self.getZ()

		raise KeyError()


	def __repr__(self):
		"""
		Returns a string representation of this instance.
		"""

		return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"


	def _assertIsSameType(self, value):
		"""
		Throws `TypeError` if `value` is not of type `Vector3DReal`.

		@param[in] value
		           The value to check.
		"""

		if not isinstance(value, Vector3DReal):
			raise TypeError()
