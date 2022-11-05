from .Base import Base

class PotentialEnergy(Base):
	"""
	Class for analysis of potential energies.
	"""

	def __init__(self, run):
		"""
		The constructor.

		@param[in] run
		           The run to analyze, as an instance of `Run`.
		"""

		super(PotentialEnergy, self).__init__(run)


	def getMPLAxesForValueAsFunctionOfTime(self):
		"""
		Returns a `matplotlib.axes.Axes` object that contains a plot of the
		potential energy, with the horizontal axis showing the simulation
		time `t`, and the vertical axis showing the potential energy at that
		point in time.
		"""

		import matplotlib.figure

		figure = matplotlib.figure.Figure()
		axes = figure.add_subplot(1, 1, 1)
		lines = []
		legendLabels = []

		data = self.getValueAsFunctionOfTime()
		_line, = axes.plot(data.keys(), data.values())
		lines.append(_line)
		legendLabels.append("Potential Energy")

		axes.legend(lines, legendLabels)

		axes.set_title("Star Polymer Potential Energy")
		axes.set_xlabel("Simulation Time t")
		axes.set_ylabel("Potential Energy")

		return axes


	def _computeValueAsFunctionOfTime(self):
		"""
		Takes the raw simulation output, and computes the potential energy as
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

		lastTimestepAnalyzed = -9e9
		while True:
			time += timestep

			particles = snapshots.readTimestep()

			if time - lastTimestepAnalyzed < self._getAnalysisTimestep():
				continue
			lastTimestepAnalyzed = time


			if particles.isEmpty():
				break

			starPolymers.setParticles(particles)

			ret[time] = starPolymers.getPotentialEnergy()

		return ret


	def _getCacheFilename(self):
		"""
		Returns the filename where cached data is (expected to be) saved to.
		"""

		return "potentialEnergy.txt"


	def _getCacheMetadata(self):
		"""
		Returns the cache metadata.
		"""

		cacheVersion = 1

		metadata = \
			{
				"cacheVersion": cacheVersion,
				"analysisTimestep": self._getAnalysisTimestep()
			}
		return metadata


	def _getAnalysisTimestep(self):
		"""
		Returns the timestep between to analysis points.
		"""

		return 100
