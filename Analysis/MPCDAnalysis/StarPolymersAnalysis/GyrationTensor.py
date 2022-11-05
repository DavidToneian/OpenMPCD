from .Base import Base

class GyrationTensor(Base):
	"""
	Class for analysis of gyration tensors.
	"""

	def __init__(self, run):
		"""
		The constructor.

		@param[in] run
		           The run to analyze, as an instance of `Run`.
		"""

		super(GyrationTensor, self).__init__(run)

	def _computeValueAsFunctionOfTime(self):
		"""
		Takes the raw simulation output, and computes the gyration tensor as
		a function of time. The result is returned as an
		`collcections.OrderedDict`, where each value is a list containing, in
		this order, the `xx`, `xy`, `xz`, `yy`, `yz`, and `zz` components of the
		gyration tensor, followed by the smallest eigenvalue and the `x`, `y`,
		and `z` components of the associated eigenvector, followed by the
		second-to-largest eigenvalue and its eigenvector, followed by the
		largest eigenvalue and its eigenvector.
		"""

		from collections import OrderedDict

		config = self.getRun().getConfiguration()
		snapshots = self._getSnapshots()
		starPolymers = self._getStarPolymers()

		ret = OrderedDict()
		timestep = config["mpc.timestep"]
		time = timestep * config["mpc.warmupSteps"]

		while True:
			time += timestep

			particles = snapshots.readTimestep()
			particles.setUniformMass(starPolymers.getParticleMass())

			if particles.isEmpty():
				break

			particles.shiftToCenterOfMassFrame()
			starPolymers.setParticles(particles)

			S = particles.getGyrationTensor()
			values = [S[0][0], S[0][1], S[0][2], S[1][1], S[1][2], S[2][2]]

			eigensystem = particles.getGyrationTensorEigensystem()
			for eigenvalue, eigenvector in eigensystem:
				values += [eigenvalue]
				values += [eigenvector[i] for i in range(0, 3)]


			ret[time] = values

		return ret


	def _getCacheFilename(self):
		"""
		Returns the filename where cached data is (expected to be) saved to.
		"""

		return "gyrationTensor.txt"


	def _getCacheMetadata(self):
		"""
		Returns the cache metadata.
		"""

		cacheVersion = 2

		metadata = {"cacheVersion": cacheVersion}
		return metadata


	def _getCacheValueInterpreter(self):
		"""
		Returns what is to be used as the `valueInterpreter` argument to
		`Cache.getDataOrderedDict`.
		"""

		def valueInterpreter(string):
			import ast
			return ast.literal_eval(string)

		return valueInterpreter
