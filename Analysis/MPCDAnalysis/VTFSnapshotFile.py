from .ParticleCollection import ParticleCollection
from .Vector3DReal import Vector3DReal

import itertools
import os.path

class VTFSnapshotFile:
	def __init__(self, path, assertReadMode = False):

		self.path = path

		self.structureBlock = None

		self._lineNumber = 0

		if os.path.exists(path):
			self.file = open(path, "r")
			self.mode = "r"

			self.readStructureBlock()
		elif os.path.exists(path + ".xz"):
			import lzma
			self.file = lzma.LZMAFile(path + ".xz", "r")
			self.mode = "r"

			self.readStructureBlock()
		else:
			if assertReadMode:
				raise RuntimeError("VTFSnapshotFile does not exist: " + path)

			self.file = open(path, "w")
			self.mode = "w"

	def isInReadMode(self):
		return self.mode == "r"

	def isInWriteMode(self):
		return self.mode == "w"

	def getStructureBlock(self):
		return self.structureBlock

	def setStructureBlock(self, structureBlock):
		self.structureBlock = structureBlock

	def readStructureBlock(self):
		if not self.isInReadMode() or self.structureBlock is not None:
			raise ValueError("")

		self.structureBlock = ""

		for line in self.file:
			self._lineNumber += 1

			if line.startswith("timestep"):
				return

			self.structureBlock += line

	def writeStructureBlock(self):
		if not self.isInWriteMode() or self.structureBlock is None:
			raise ValueError("")

		self.file.write(self.structureBlock)

	def readTimestep(self):
		positions = []
		velocities = []
		for line in self.file:
			self._lineNumber += 1

			if line.startswith("timestep"):
				break

			parts = line.split()

			if len(parts) not in [3, 6]:
				message = "Unexpected line, number " + str(self._lineNumber)
				message += ":\n"
				message += line
				raise ValueError(message)

			x = float(parts[0])
			y = float(parts[1])
			z = float(parts[2])
			positions.append(Vector3DReal(x, y, z))

			if len(parts) == 6:
				vx = float(parts[3])
				vy = float(parts[4])
				vz = float(parts[5])
				velocities.append(Vector3DReal(vx, vy, vz))

		collection = ParticleCollection()
		collection.setPositionsAndVelocities(positions, velocities)

		return collection

	def writeTimestep(self, particleCollection):
		if not self.isInWriteMode():
			raise ValueError("")

		positions = particleCollection.getPositions()
		velocities = particleCollection.getVelocities()

		self.file.write("timestep\n")
		for position, velocity in itertools.izip_longest(positions, velocities):
			self.file.write(str(position.getX().real) + " ")
			self.file.write(str(position.getY().real) + " ")
			self.file.write(str(position.getZ().real))

			if velocity is not None:
				self.file.write(" " + str(velocity.getX().real))
				self.file.write(" " + str(velocity.getY().real))
				self.file.write(" " + str(velocity.getZ().real))

			self.file.write("\n")
