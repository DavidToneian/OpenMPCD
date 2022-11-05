from .Base import Base

class EckartAngularVelocityVector(Base):
	"""
	Class for analysis of Eckart-frame angular velocity vectors, as defined in
	`MPCDAnalysis.EckartSystem`.
	"""

	def __init__(self, run):
		"""
		The constructor.

		@param[in] run
		           The run to analyze, as an instance of `Run`.
		"""

		super(EckartAngularVelocityVector, self).__init__(run)


	def getMPLAxesForValueAsFunctionOfTime(self):
		"""
		Returns a `matplotlib.axes.Axes` object that contains a plot of the
		Eckart angular velocities of star polymer, with the horizontal axis
		showing the simulation time `t`, and the Cartesian components of the
		Eckart angular velocity vector, in radians per time unit.
		"""

		import matplotlib.figure

		figure = matplotlib.figure.Figure()
		axes = figure.add_subplot(1, 1, 1)
		lines = []
		legendLabels = []

		data = self.getValueAsFunctionOfTime()
		frequencies = [[], [], []]
		for datum in data.values():
			for i in [0, 1, 2]:
				frequencies[i].append(datum[i])

		_line, = axes.plot(data.keys(), frequencies[0])
		lines.append(_line)
		legendLabels.append("Eckart Angular Velocity: x Component")

		_line, = axes.plot(data.keys(), frequencies[1])
		lines.append(_line)
		legendLabels.append("Eckart Angular Velocity: y Component")

		_line, = axes.plot(data.keys(), frequencies[2])
		lines.append(_line)
		legendLabels.append("Eckart Angular Velocity: z Component")

		axes.legend(lines, legendLabels)

		axes.set_title("Star Polymer Eckart Angular Velocity")
		axes.set_xlabel("Simulation Time t")
		axes.set_ylabel("Eckart Angular Velocity [rad/T]")

		return axes


	def _computeValueAsFunctionOfTime(self):
		"""
		Takes the raw simulation output, and computes the Eckart angular
		velocity vector as a function of time. The result is returned as an
		`collcections.OrderedDict`.
		"""

		from collections import OrderedDict


		config = self.getRun().getConfiguration()
		snapshots = self._getSnapshots()
		starPolymers = self._getStarPolymers()

		ret = OrderedDict()
		timestep = config["mpc.timestep"]
		time = timestep * config["mpc.warmupSteps"]

		eckartSystem = None
		while True:
			time += timestep

			particles = snapshots.readTimestep()
			particles.setUniformMass(starPolymers.getParticleMass())

			if particles.isEmpty():
				break

			if eckartSystem is None:
				from MPCDAnalysis.EckartSystem import EckartSystem
				eckartSystem = EckartSystem(particles)

			ret[time] = eckartSystem.getEckartAngularVelocityVector(particles)

		return ret


	def _getCacheFilename(self):
		"""
		Returns the filename where cached data is (expected to be) saved to.
		"""

		return "eckartAngularVelocityVectors.txt"


	def _getCacheValueInterpreter(self):
		"""
		Returns what is to be used as the `valueInterpreter` argument to
		`Cache.getDataOrderedDict`.
		"""

		def valueInterpreter(string):
			import ast
			return ast.literal_eval(string)

		return valueInterpreter
