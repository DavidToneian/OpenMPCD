class FlowProfile:
	"""
	Represents a flow profile, as created by `OpenMPCD::FlowProfile`.
	"""

	def __init__(self, rundir):
		"""
		The constructor.
		"""

		self._setConfigOrCheckCompatibility(rundir)

		self._lastRun = None

		self._points = {}
		self._outputBlocks = []

		self.addRun(rundir)


	def addRun(self, rundir):
		"""
		Adds the flow profile data in the given `rundir` to the data gathered so
		far.
		"""

		if self._outputBlocks:
			raise Exception("Cannot currently concatenate two multi-block runs")


		from .Run import Run
		self._lastRun = Run(rundir)


		self._setConfigOrCheckCompatibility(rundir)

		import os.path

		if os.path.isfile(rundir + '/flowProfile.data.bz2'):
			import bz2
			tmpfile = bz2.BZ2File(rundir + '/flowProfile.data.bz2')
		elif os.path.isfile(rundir + '/flowProfile.data'):
			tmpfile = open(rundir + '/flowProfile.data', 'r')
		else:
			raise Exception("Invalid rundir: " + rundir)

		for line in tmpfile:
			if line[0] == '#':
				continue

			if line == "\n":
				self._outputBlocks.append(self._points)
				self._points = {}
				continue

			columns = line.split()
			slice_ = [ int(i) for i in columns[0:3] ]
			mean = [ float(i) for i in columns[3:6] ]
			standardDeviation = [ float(i) for i in columns [6:9] ]
			sampleSize = int(columns[9])

			self._addSampleToPoint(slice_, mean, standardDeviation, sampleSize)

		if self._outputBlocks:
			self._outputBlocks.append(self._points)
			self._points = {}


	def getMPLAxesForFlowAlongXAsFunctionOfY(
		self, standardError = 1, theory = True, shift = True):
		"""
		Returns a `matplotlib.axes.Axes` object that contains a plot of the
		flow profile, with the horizontal axis showing the simulation `y`
		coordinate, and the vertical axis showing the average flow velocity in
		`x` direction (along with its standard deviation) at that point.

		@param[in] standardError
		           Include a shaded region around the mean that corresponds to
		           `standardError` times the standard error. Set to `0` to omit
		           this shaded region altogether.
		@param[in] theory
		           Set to `True` to also include a plot of the theoretical shear
		           flow profile.
		@param[in] shift
		           Set to `True` to shift the data points along the `y`
		           direction in such a way that a data point lies not at the
		           beginning of the `y` segment it describes, but rather at its
		           center.
		"""

		import matplotlib.figure

		figure = matplotlib.figure.Figure()
		axes = figure.add_subplot(1, 1, 1)
		lines = []
		legendLabels = []

		velocityIndex = 0

		data = self.getFlowProfileAsFunctionOfY(shift)
		values = \
			[velocity[velocityIndex].getSampleMean()
			 for velocity in data.values()]
		_line, = axes.plot(data.keys(), values)
		lines.append(_line)
		legendLabels.append("Flow Profile")

		if theory:
			theoryData = \
				self.getAnalyticShearFlowProfileAsFunctionOfY(shift = shift)
			_line, = axes.plot(theoryData.keys(), theoryData.values())
			lines.append(_line)
			legendLabels.append("Theory")

		if standardError != 0:
			if standardError < 0:
				raise Exception()

			import math
			import numpy

			values = numpy.array(values)
			sqrt_n = math.sqrt(velocity[velocityIndex].getSampleSize())
			errors = \
				numpy.array(
					[velocity[velocityIndex].getSampleStandardDeviation() /
					 sqrt_n
					 for velocity in data.values()])

			axes.fill_between(
				data.keys(),
				values - standardError * errors,
				values + standardError * errors,
				alpha = 0.2)

		axes.legend(lines, legendLabels)

		return axes


	def getFlowProfileAsFunctionOfY(self, shift = True):
		"""
		Returns the flow profile as a function of `y`.

		The value returned is a dictionary, with the keys being `y` coordinates
		in simulation space, and the values being a lists of three
		`OnTheFlyStatistics` instances, for the flow velocities along the `x`,
		`y`, and `z` direction, respectively.

		@param[in] shift
		           Set to `True` to shift the data points along the `y`
		           direction in such a way that a data point lies not at the
		           beginning of the `y` segment it describes, but rather at its
		           center.
		"""

		from .OnTheFlyStatistics import OnTheFlyStatistics
		from collections import OrderedDict

		ret = OrderedDict()

		for _, xval in self._points.items():
			for yIndex, yval in xval.items():
				yCoord = float(yIndex) / self._linearSubdivisions
				if shift:
					yCoord += 0.5 / self._linearSubdivisions
				if yCoord not in ret:
					ret[yCoord] = \
						[
							OnTheFlyStatistics(),
							OnTheFlyStatistics(),
							OnTheFlyStatistics()
						]

				for _, zval in yval.items():
					ret[yCoord][0].mergeSample(zval[0])
					ret[yCoord][1].mergeSample(zval[1])
					ret[yCoord][2].mergeSample(zval[2])

		return ret


	def getGlobalFlowStatistics(self):
		"""
		Returns a list of three `OnTheFlyStatistics` objects that combine, one
		for each of the three Cartesian coordinates, that combines all samples
		of particle velocities across the simulation volume.
		"""

		from .OnTheFlyStatistics import OnTheFlyStatistics

		ret = [OnTheFlyStatistics(), OnTheFlyStatistics(), OnTheFlyStatistics()]

		for xval in self._points.values():
			for yval in xval.values():
				for zval in yval.values():
					for i in [0, 1, 2]:
						ret[i].mergeSample(zval[i])

		return ret



	def getAnalyticShearFlowProfileAsFunctionOfY(
		self, shearRate = None, shift = True):
		"""
		Returns the analytic flow profile of a shear flow as a function of `y`.

		The shear flow is assumed to have the flow direction along the positive
		`x` direction, and the gradient direction is assumed to be the `y`
		direction.

		The value returned is a dictionary, with the keys being `y` coordinates
		in simulation space, and the values being the mean flow speed along the
		`x` direction.

		@param[in] shearRate
		           Sets the shear rate. If `None`, and a run has been added
		           which does specify a shear rate, that value is assumed.
		           Otherwise, an exception is thrown.

		@param[in] shift
		           Set to `True` to shift the data points along the `y`
		           direction in such a way that a data point lies not at the
		           beginning of the `y` segment it describes, but rather at its
		           center.
		"""

		if shearRate is None:
			if not hasattr(self, "_shearRate"):
				raise Exception()
			shearRate = self._shearRate

		if not isinstance(shearRate, float):
			raise Exception()

		if not hasattr(self, "_simBoxSizeY"):
			raise Exception()
		simBoxSizeY = self._simBoxSizeY

		if not hasattr(self, "_linearSubdivisions"):
			raise Exception()
		linearSubdivisions = self._linearSubdivisions

		from collections import OrderedDict

		ret = OrderedDict()

		maxYIndex = simBoxSizeY * linearSubdivisions - 1
		for yIndex in range(0, maxYIndex + 1):
			yCoord = float(yIndex) / linearSubdivisions
			if shift:
				yCoord += 0.5 / linearSubdivisions

			v_x = (yCoord - 0.5 * simBoxSizeY) * shearRate
			ret[yCoord] = v_x

		return ret


	def showVectorFieldGUI(self):
		import matplotlib.pyplot
		from .PlotTools import DiscreteSliderWidget


		fig, ax = matplotlib.pyplot.subplots()
		ax.set_title("Flow Profile")
		ax.set_xticks(range(0, self._simBoxSizeX + 1))
		ax.set_yticks(range(0, self._simBoxSizeY + 1))
		ax.grid(True, which = 'both')
		matplotlib.pyplot.xlabel("x")
		matplotlib.pyplot.ylabel("y")
		ax.set_xlim([0, self._simBoxSizeX])
		ax.set_ylim([0, self._simBoxSizeY])
		matplotlib.pyplot.subplots_adjust(left = 0.1, bottom = 0.25)


		currentParameters = \
			{
				'outputBlockID': 0,
				'z': 0,
			}

		timeAxes = matplotlib.pyplot.axes([0.1, 0.1, 0.8, 0.03])
		timeSlider = \
			DiscreteSliderWidget(
				timeAxes, "Output Block",
				currentParameters['outputBlockID'], len(self._outputBlocks) - 1,
				valinit = 0, valfmt = '%0.0f')

		zAxes = matplotlib.pyplot.axes([0.1, 0.05, 0.8, 0.03])
		zSlider = \
			DiscreteSliderWidget(
				zAxes, "z",
				currentParameters['z'], self._simBoxSizeZ - 1,
				valinit = 0, valfmt = '%0.0f')

		xPoints = [x + 0.5 for x in self._outputBlocks[0].keys()]
		yPoints = [y + 0.5 for y in self._outputBlocks[1].keys()]

		import numpy
		X, Y = numpy.meshgrid(xPoints, yPoints)

		quiverProxy = []

		def updateData():
			outputBlock = self._outputBlocks[currentParameters['outputBlockID']]

			U = []
			V = []
			C = []
			z = currentParameters['z']
			for yIdx, y in enumerate(self._outputBlocks[1].keys()):
				for x in self._outputBlocks[0].keys():
					v = outputBlock[x][y][z]
					v = [v[i].getSampleMean() for i in range(0, 3)]

					if yIdx == len(U):
						U.append([])
						V.append([])
						C.append([])

					U[yIdx].append(v[0])
					V[yIdx].append(v[1])
					C[yIdx].append(v[2])

			if quiverProxy:
				quiverProxy[0].set_UVC(U, V, C)
			else:
				quiverProxy.append(ax.quiver(X, Y, U, V, C, units = 'width'))

			fig.canvas.draw_idle()

		def updateData_time(newOutputBlockID):
			currentParameters['outputBlockID'] = newOutputBlockID
			updateData()

		def updateData_z(newZ):
			currentParameters['z'] = newZ
			updateData()

		timeSlider.on_changed(updateData_time)
		zSlider.on_changed(updateData_z)


		updateData()

		mng = matplotlib.pyplot.get_current_fig_manager()
		mng.resize(*mng.window.maxsize())
		matplotlib.pyplot.show()


	def _setConfigOrCheckCompatibility(self, rundir):
		"""
		Reads the configuration in the given `rundir`, and either sets it to be
		the reference configuration if this is the first `rundir` this instance
		is supplied, or otherwise verifies that the configurations are
		compatible.
		"""

		from .Configuration import Configuration
		config = Configuration(rundir)

		if not hasattr(self, "_linearSubdivisions"):
			attributeList = \
				[
					"_linearSubdivisions", "_shearRate",
					"_simBoxSizeX", "_simBoxSizeY", "_simBoxSizeZ",
				]
			for attribute in attributeList:
				if hasattr(self, attribute):
					raise Exception()

			self._linearSubdivisions = 1

			if "instrumentation.flowProfile.cellSubdivision.y" in config:
				value = config["instrumentation.flowProfile.cellSubdivision.y"]
				self._linearSubdivisions = value

			if "boundaryConditions.LeesEdwards.shearRate" in config:
				self._shearRate = \
					config["boundaryConditions.LeesEdwards.shearRate"]
			else:
				self._shearRate = 0.0

			self._simBoxSizeX = config["mpc.simulationBoxSize.x"]
			self._simBoxSizeY = config["mpc.simulationBoxSize.y"]
			self._simBoxSizeZ = config["mpc.simulationBoxSize.z"]

			return

		linearSubdivisions = 1
		if "instrumentation.flowProfile.cellSubdivision.y" in config:
			linearSubdivisions = \
				config["instrumentation.flowProfile.cellSubdivision.y"]

		if self._linearSubdivisions != linearSubdivisions:
			raise Exception()

		newShearRate = 0.0
		if "boundaryConditions.LeesEdwards.shearRate" in config:
				newShearRate = \
					config["boundaryConditions.LeesEdwards.shearRate"]
		if self._shearRate != newShearRate:
			raise Exception()

		if self._simBoxSizeY != config["mpc.simulationBoxSize.y"]:
			raise Exception()



	def _addSampleToPoint(self, slice_, means, standardDeviations, sampleSize):
		"""
		Adds the data given (`means`, `standardDeviations`, and `sampleSize`) to
		the point specified by the slice coordinates in `slice_`, which is
		assumed to be a list of three coordinates (each corresponding to the
		simulation coordinates after being multiplied with the appropriate
		`instrumentation.cellSubdivision` configuration value).

		If no data exists yet for that point, a new instance of
		`OnTheFlyStatistics` is created.
		"""

		from .OnTheFlyStatistics import OnTheFlyStatistics
		from collections import OrderedDict

		x, y, z = slice_
		if x not in self._points:
			self._points[x] = OrderedDict()
		if y not in self._points[x]:
			self._points[x][y] = OrderedDict()
		if z not in self._points[x][y]:
			self._points[x][y][z] = []
			self._points[x][y][z].append(OnTheFlyStatistics())
			self._points[x][y][z].append(OnTheFlyStatistics())
			self._points[x][y][z].append(OnTheFlyStatistics())

		newSampleX = \
			OnTheFlyStatistics(means[0], standardDeviations[0] ** 2, sampleSize)
		newSampleY = \
			OnTheFlyStatistics(means[1], standardDeviations[1] ** 2, sampleSize)
		newSampleZ = \
			OnTheFlyStatistics(means[2], standardDeviations[2] ** 2, sampleSize)

		self._points[x][y][z][0].mergeSample(newSampleX)
		self._points[x][y][z][1].mergeSample(newSampleY)
		self._points[x][y][z][2].mergeSample(newSampleZ)
