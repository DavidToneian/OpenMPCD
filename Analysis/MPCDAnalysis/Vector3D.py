import math

class Vector3D:
	def __init__(self, x, y, z):
		self.x = complex(x)
		self.y = complex(y)
		self.z = complex(z)

	def getX(self):
		return self.x

	def getY(self):
		return self.y

	def getZ(self):
		return self.z

	def getRealPart(self):
		ret = Vector3D(0, 0, 0)
		ret.x = self.x.real
		ret.y = self.y.real
		ret.z = self.z.real

		return ret

	def getComplexConjugate(self):
		return Vector3D(self.x.conjugate(), self.y.conjugate(), self.z.conjugate())

	def dot(self, rhs):
		return self.x.conjugate() * rhs.x + self.y.conjugate() * rhs.y + self.z.conjugate() * rhs.z

	def cross(self, rhs):
		cx = self.y * rhs.z - self.z * rhs.y;
		cy = self.z * rhs.x - self.x * rhs.z;
		cz = self.x * rhs.y - self.y * rhs.x;
		return Vector3D(cx, cy, cz)

	def getLengthSquared(self):
		return self.dot(self).real

	def getLength(self):
		return math.sqrt(self.getLengthSquared())

	def getNormalized(self):
		factor = 1.0 / self.getLength()
		return Vector3D(self.x * factor, self.y * factor, self.z * factor)

	def normalize(self):
		factor = 1.0 / self.getLength()
		self.x = self.x * factor
		self.y = self.y * factor
		self.z = self.z * factor

	def getProjectionOnto(self, rhs):
		normalized = rhs.getNormalized()
		return normalized * normalized.dot(self)

	def getPerpendicularTo(self, rhs):
		return self -self.getProjectionOnto(rhs)

	def getRotatedAroundNormalizedAxis(self, axis, angle):
		thisDotAxis = self.dot(axis)
		axisCrossThis = axis.cross(self)
		projectionOntoAxis = axis * thisDotAxis
		return \
			projectionOntoAxis + \
			(self -projectionOntoAxis) * math.cos(angle) + \
			axisCrossThis * math.sin(angle)

	def rotateAroundNormalizedAxis(self, axis, angle):
		rotated = self.getRotatedAroundNormalizedAxis(axis, angle)
		self.x = rotated.x
		self.y = rotated.y
		self.z = rotated.z

	def __eq__(self, rhs):
		if self.x != rhs.x:
			return False
		if self.y != rhs.y:
			return False
		if self.z != rhs.z:
			return False
		return True

	def __add__(self, rhs):
		if not isinstance(rhs, Vector3D):
			return NotImplemented

		return Vector3D(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)

	def __sub__(self, rhs):
		if not isinstance(rhs, Vector3D):
			return NotImplemented

		return Vector3D(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)

	def __mul__(self, rhs):
		if isinstance(rhs, float) or isinstance(rhs, int) or isinstance(rhs, complex):
			return Vector3D(self.x * rhs, self.y * rhs, self.z * rhs)
		else:
			return NotImplemented

	def __div__(self, rhs):
		if isinstance(rhs, float) or isinstance(rhs, int) or isinstance(rhs, complex):
			return Vector3D(self.x / rhs, self.y / rhs, self.z / rhs)
		else:
			return NotImplemented

	def __getitem__(self, index):
		if index == 0:
			return self.getX()

		if index == 1:
			return self.getY()

		if index == 2:
			return self.getZ()

		raise ValueError("invalid index")

	def __str__(self):
		return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"

	def __repr__(self):
		return self.__str__()
