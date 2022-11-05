def mergeAxes(axesList):
	"""
	Takes the given `axesList`, and returns a new `matplotlib.axes` instance
	that contains all the lines in the given axes.

	@param[in] axesList
	           A list of instances of `matplotlib.axes`.
	"""

	import matplotlib

	if not isinstance(axesList, list):
		raise TypeError()
	for axes in axesList:
		if not isinstance(axes, matplotlib.axes.Axes):
			raise TypeError()


	retAxes = matplotlib.figure.Figure().add_subplot(1, 1, 1)
	titles = []
	xLabels = []
	yLabels = []
	retLines = []
	legendLabels = []

	for sourceAxes in axesList:
		for line in sourceAxes.get_lines():
			newLine, = retAxes.plot(line.get_xdata(), line.get_ydata())
			retLines.append(newLine)

		titles.append(sourceAxes.get_title())
		xLabels.append(sourceAxes.get_xlabel())
		yLabels.append(sourceAxes.get_ylabel())

		for text in sourceAxes.get_legend().get_texts():
			legendLabels.append(text.get_text())

	retAxes.legend(retLines, legendLabels)

	uniqueTitles = list(set(titles))
	uniqueXLabels = list(set(xLabels))
	uniqueYLabels = list(set(yLabels))

	def labelSetter(uniqueLabels, setter):
		if len(uniqueLabels) == 0:
			return

		if len(uniqueLabels) == 1:
			setter(uniqueLabels[0])
			return

		label = None
		for l in uniqueLabels:
			if label is None:
				label = l
			else:
				label += " | " + l
		setter(label)

	labelSetter(uniqueTitles, retAxes.set_title)
	labelSetter(uniqueXLabels, retAxes.set_xlabel)
	labelSetter(uniqueYLabels, retAxes.set_ylabel)

	return retAxes
