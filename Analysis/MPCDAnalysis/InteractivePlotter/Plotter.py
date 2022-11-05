from .App import App

from .. import Exceptions
from .. import PlotTools
from .. import Utilities

import copy
import os

class Plotter:
	"""
	Graphical User Interface for plotting data.

	@var _plotDashedIfNegative  Whether the negative parts of plots should instead be plotted
	                            positively, but with dashed lines.
	@var _yScaleIsLog           Whether the y axis is logarithmic.
	@var _plotSelector          Instance of _PlotSelectionList used to select plots for display.
	@var _hiddenCharacteristics List of plot characteristics that are hidden in the plot selector.
	@var _characteristicsOrder  Order of the plot characteristics.
	"""

	def __init__(self):
		self.app = App(self)
		self.plots = {}
		self.nextPlotID = 0
		self._plotDashedIfNegative = False
		self._yScaleIsLog = False
		self._hiddenCharacteristics = []
		self._characteristicsOrder = []

	def run(self):
		self.app.MainLoop()

	def provideData(self, x, y, show=True, label=None, characteristics=None):
		"""
		Provides a new plot by specifying the x and y coordinates.

		@param[in] x               A list of x coordinates.
		@param[in] y               A list of y coordinates corresponding to the x coordinates.
		@param[in] show            Whether to draw the plot initially, as a bool.
		@param[in] label           The label for the plot.
		@param[in] characteristics Plot characteristics, as used in the plot selection window.
		@throw TypeError Throws if show is not of type bool.
		"""

		if type(show) is not bool:
			raise TypeError("Inavlid type for parameter 'show'")

		lines = None
		if show:
			lines = self.plot(x, y, label=label)
			self.app.canvasFrame.update()

		if characteristics is None:
			characteristics = {}

		self.plots[self.nextPlotID] = {'lines': lines, 'x': x, 'y': y, 'show': show, 'label': label, 'characteristics': characteristics}
		self.nextPlotID = self.nextPlotID + 1

		self._updatePlotSelector()

	def provideDataFile(self, path, characteristics={}):
		"""
		Adds the given file, or rather its contents, to the list of available plots.

		The added plot is not drawn on the canvas.

		Each line in the file that is of the form
		#name = value
		is considered to describe the plot's characteristic "name", the value of which
		is "value". "value" is treated as a float, if possible, and as a string otherwise.

		Any other line must consist of a float, followed by a space, and another float.
		The first float corresponds to the x coordinate of a plot point, the second
		float is the corresponding y coordinate.

		@param[in] path            The path to the file.
		@param[in] characteristics A dictionary of characteristics for the plot.
		                           Conflicting file comments are ignored.
		@throw FileFormatException Throws if the given file did not adhere to the expected format.
		"""

		x = []
		y = []

		with Utilities.openPossiblyCompressedFile(path) as file:
			for line in file:
				if line[0] == '#':
					parts = line.split('=')
					if len(parts) > 0 and parts[0][1:].strip() == 'rundirs':
						parts = line.split('=', 1)

					if len(parts) != 2:
						raise Exceptions.FileFormatException("Unknown comment format in " + path)

					property_, value = parts
					property_ = property_[1:].strip()

					if property_ in characteristics:
						continue

					try:
						value = float(value)
					except:
						pass
					characteristics[property_] = value

					continue

				parts = line.split()
				if len(parts) != 2:
					raise Exceptions.FileFormatException("Unknown value format in " + path)

				x.append(float(parts[0]))
				y.append(float(parts[1]))

		self.provideData(x, y, show=False, characteristics=characteristics)

	def provideDataDirectory(self, path, characteristics={}, recursive=False):
		"""
		Adds all files in the given directory to the available data files.

		Each file corresponds to a plot.
		The added plots are not drawn on the canvas.

		@param[in] path            The path to the directory to be added.
		@param[in] characteristics A dictionary of characteristics for all plots.
		                           Conflicting file comments are ignored.
		@param[in] recursive       Whether to traverse the directory recursively.
		@throw FileFormatException Any of the given files did not adhere to the expected format,
		                           as described in provideDataFile.
		"""

		for file in os.listdir(path):
			filepath = path + '/' + file
			if os.path.isfile(filepath):
				self.provideDataFile(filepath, copy.copy(characteristics))
			else:
				if recursive:
					self.provideDataDirectory(filepath, characteristics, recursive)

	def setXScale(self, scale):
		self.app.canvasFrame.axes.set_xscale(scale)
		self.app.canvasFrame.setLogXCheckbox(scale == 'log')

	def setYScale(self, scale):
		self.app.canvasFrame.axes.set_yscale(scale)
		self.app.canvasFrame.setLogYCheckbox(scale == 'log')
		if scale == 'log':
			self._yScaleIsLog = True
		else:
			self._yScaleIsLog = False

	def setPlotDashedIfNegative(self, state):
		self._plotDashedIfNegative = state
		self.app.canvasFrame.setDashedIfNegativeCheckbox(state)
		self.replot()

	def hideCharacteristics(self, toHide):
		"""
		Hides the given characteristics from the plot selection window.

		@param[in] toHide A list of plot characteristics to hide.
		"""

		self._hiddenCharacteristics = self._hiddenCharacteristics + toHide
		self._updatePlotSelector()

	def setCharacteristicsOrder(self, order):
		"""
		Sets the order in which plot characteristics are displayed in the plot selection window.

		@param[in] order An ordered list of characteristics to display. Characteristics not in
		                 this list will be displayed in undetermined order after all
		                 characteristics present in this list.
		"""

		self._characteristicsOrder = [x for x in order]

	def setSortingCharacteristic(self, characteristic, ascending):
		"""
		Sets by which characteristic to sort the plots in the plot selection window.

		@param[in] characteristic The characteristic name to sort by.
		@param[in] ascending      Set to true to sort by ascending values, false for descending.
		@throw InvalidArgumentException Throws if there is no such characteristic.
		"""

		columnID = None

		for column in range(0, self._plotSelector.GetColumnCount()):
			name = self._plotSelector.getColumnName(column)
			if name == characteristic:
				columnID = column
				break

		if columnID is None:
			raise Exceptions.InvalidArgumentException('No such characteristic.')

		self._plotSelector.SortListItems(columnID, ascending)

	def toggleVisibility(self, plotID):
		if not self.plots[plotID]['show']:
			lines = self.plot(self.plots[plotID]['x'], self.plots[plotID]['y'], label=self.plots[plotID]['label'])
			self.plots[plotID]['lines'] = lines
			self.plots[plotID]['show'] = True
		else:
			for line in self.plots[plotID]['lines']:
				self.app.canvasFrame.axes.lines.remove(line)
			self.plots[plotID]['lines'] = None
			self.plots[plotID]['show'] = False

		self.app.canvasFrame.update()
		self._updatePlotSelector()

	def replot(self):
		for line in self.app.canvasFrame.axes.get_lines():
			self.app.canvasFrame.axes.lines.remove(line)

		self.app.canvasFrame.axes.set_color_cycle(None)
		for plotID, plot in self.plots.items():
			lines = None
			if plot['show']:
				lines = self.plot(plot['x'], plot['y'], label=plot['label'])

			plot['lines'] = lines

	def plot(self, x, y, **kwargs):
		if self._plotDashedIfNegative:
			lines = PlotTools.plotDashedIfNegative(x, y, logY=self._yScaleIsLog, ax=self.app.canvasFrame.axes, **kwargs)
		else:
			lines = self.app.canvasFrame.axes.plot(x, y, **kwargs)

		return lines

	def _updatePlotSelector(self):
		"""
		Updates the plot selector.
		"""

		characteristics = self._getListOfPlotCharacteristics(['Plotted'])

		sortState = None
		if self._plotSelector.GetColumnCount() != 0:
			sortState = self._plotSelector.GetSortState()
			if self._plotSelector.getColumnName(sortState[0]) in self._hiddenCharacteristics:
				sortState = None

		columnWidths = {}
		for column in range(0, self._plotSelector.GetColumnCount()):
			name = self._plotSelector.getColumnName(column)
			columnWidths[name] = self._plotSelector.GetColumnWidth(column)

		self._plotSelector.DeleteAllColumns()

		column = 0
		for characteristic in characteristics:
			if characteristic in self._hiddenCharacteristics:
				continue

			self._plotSelector.InsertColumn(column, characteristic)
			if characteristic in columnWidths:
				self._plotSelector.SetColumnWidth(column, columnWidths[characteristic])
			column = column + 1

		self._plotSelector.SetColumnCount(len(characteristics))

		plotDict = {}
		for plotID, plot in self.plots.items():
			plotCharacteristics = []

			for characteristic in characteristics:
				if characteristic in self._hiddenCharacteristics:
					continue

				if characteristic == 'Plotted':
					plotCharacteristics.append(plot['show'])
					continue

				if characteristic in plot['characteristics']:
					plotCharacteristics.append(plot['characteristics'][characteristic])
				else:
					plotCharacteristics.append(None)

			plotDict[plotID] = plotCharacteristics

		self._plotSelector.itemDataMap = plotDict
		self._plotSelector.itemIndexMap = plotDict.keys()
		self._plotSelector.SetItemCount(len(plotDict))

		if sortState is not None:
			self._plotSelector.SortListItems(sortState[0], sortState[1])


	def _getListOfPlotCharacteristics(self, initial=[]):
		"""
		Returns a list of characteristics that are used in any of the known plots.

		@param[in] initial A list of characteristics that are to be included in the returned result,
		                   even if not used by any plot.
		"""

		characteristics = []

		for c in self._characteristicsOrder:
			if c in characteristics:
				continue

			if c in initial:
				characteristics.append(c)
				continue

			for _plotID, plot in self.plots.items():
				if c in plot['characteristics'].keys():
					characteristics.append(c)
					break


		for c in initial:
			if c not in characteristics:
				characteristics.append(c)

		for _plotID, plot in self.plots.items():
			for c in plot['characteristics'].keys():
				if c not in characteristics:
					characteristics.append(c)

		return characteristics

	def _registerPlotSelector(self, plotSelector):
		"""
		Registers an instance of _PlotSelectionList

		@param[in] plotSelector The instance to register.
		"""

		self._plotSelector = plotSelector
