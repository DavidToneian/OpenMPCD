from .Base import Base

class RadiusOfGyration(Base):
	"""
	Class for analysis of radii of gyration.
	"""

	def __init__(self, run):
		"""
		The constructor.

		@param[in] run
		           The run to analyze, as an instance of `Run`.
		"""

		super(RadiusOfGyration, self).__init__(run)


	def getMPLAxesForValueAsFunctionOfTime(self):
		"""
		Returns a `matplotlib.axes.Axes` object that contains a plot of the
		radius of gyration, with the horizontal axis showing the simulation
		time `t`, and the vertical axis showing the radius of gyration at that
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
		legendLabels.append("Radius of Gyration")

		axes.legend(lines, legendLabels)

		axes.set_title("Star Polymer Radius of Gyration")
		axes.set_xlabel("Simulation Time t")
		axes.set_ylabel("Radius of Gyration")

		return axes


	def _computeValueAsFunctionOfTime(self):
		"""
		Takes the raw simulation output, and computes the radius of gyration as
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

			ret[time] = particles.getRadiusOfGyration()

		return ret


	def _getCacheFilename(self):
		"""
		Returns the filename where cached data is (expected to be) saved to.
		"""

		return "radiusOfGyration.txt"
