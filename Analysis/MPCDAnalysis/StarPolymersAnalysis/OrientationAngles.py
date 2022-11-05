from .Base import Base

class OrientationAngles(Base):
	"""
	Class for analysis of the orientation of the eigenvectors of the gyration
	tensor, relative to the flow direction.
	"""

	def __init__(self, run):
		"""
		The constructor.

		@param[in] run
		           The run to analyze, as an instance of `Run`.
		"""

		super(OrientationAngles, self).__init__(run)


	def getMPLAxesForValueAsFunctionOfTime(self):
		"""
		Returns a `matplotlib.axes.Axes` object that contains a plot of the
		orientation angles of the three eigenvectors of the gyration tensor with
		the flow direction, with the horizontal axis showing the simulation
		time `t`, and the vertical axis showing the angles in radians.
		"""

		import matplotlib.figure

		figure = matplotlib.figure.Figure()
		axes = figure.add_subplot(1, 1, 1)
		lines = []
		legendLabels = []

		data = self.getValueAsFunctionOfTime()
		angles = [[], [], []]
		for datum in data.values():
			for i in [0, 1, 2]:
				angles[i].append(datum[i][1])

		_line, = axes.plot(data.keys(), angles[0])
		lines.append(_line)
		legendLabels.append("Orientation Angle: Smallest Eigenvalue")

		_line, = axes.plot(data.keys(), angles[1])
		lines.append(_line)
		legendLabels.append("Orientation Angle: Middle Eigenvalue")

		_line, = axes.plot(data.keys(), angles[2])
		lines.append(_line)
		legendLabels.append("Orientation Angle: Largest Eigenvalue")

		axes.legend(lines, legendLabels)

		axes.set_title("Star Polymer Orientation Angles")
		axes.set_xlabel("Simulation Time t")
		axes.set_ylabel("Orientation Angles [rad]")

		return axes


	def _computeValueAsFunctionOfTime(self):
		"""
		Takes the raw simulation output, and computes the orientation angles as
		a function of time. The result is returned as an
		`collcections.OrderedDict`.
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

			ret[time] = particles.getOrientationAngles(self._getFlowDirection())

		return ret


	def _getCacheFilename(self):
		"""
		Returns the filename where cached data is (expected to be) saved to.
		"""

		return "orientationAngles.txt"


	def _getCacheMetadata(self):
		"""
		Returns the cache metadata.
		"""

		cacheVersion = 2

		metadata = \
			{
				"cacheVersion": cacheVersion,
				"flowDirection": self._getFlowDirection()
			}
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


	def _getFlowDirection(self):
		"""
		Returns the flow direction of shear flow.
		"""

		return [1.0, 0.0, 0.0]
