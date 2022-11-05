from .PlotSelectionFrame import PlotSelectionFrame

import matplotlib.backends.backend_wx
import matplotlib.backends.backend_wxagg
import matplotlib.figure
import wx

class CanvasFrame(wx.Frame):
	"""
	Main window, containing the plotting canvas.
	"""

	def __init__(self, plotter):
		sizeX = 1200
		sizeY = 780
		dpi = 100
		wx.Frame.__init__(self, None, title='my title', size=(sizeX, sizeY))

		self.plotter = plotter

		self.mainSizer = wx.BoxSizer(wx.VERTICAL)
		self.SetSizer(self.mainSizer)

		self.canvasHorizontalSizer = wx.BoxSizer(wx.HORIZONTAL)
		self.mainSizer.Add(self.canvasHorizontalSizer, 1, wx.LEFT | wx.TOP | wx.EXPAND)

		self.figure = matplotlib.figure.Figure(figsize=(sizeX / dpi, sizeY / dpi), dpi=dpi)
		self.canvas = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(self, wx.ID_ANY, self.figure)
		self.canvasHorizontalSizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.EXPAND)



		self.plotOptions = {}

		self.plotOptions['sizer'] = wx.BoxSizer(wx.HORIZONTAL)
		self.mainSizer.Add(self.plotOptions['sizer'], 0, wx.LEFT | wx.EXPAND)


		self.plotOptions['checkBoxes'] = {}

		self.plotOptions['checkBoxes']['logx'] = wx.CheckBox(self)
		self.plotOptions['sizer'].Add(self.plotOptions['checkBoxes']['logx'])
		self.Bind(wx.EVT_CHECKBOX, self.onPlotOptionsCheckbox, self.plotOptions['checkBoxes']['logx'])
		self.plotOptions['sizer'].Add(wx.StaticText(self, label="log-x"), 0, wx.LEFT | wx.EXPAND)

		self.plotOptions['checkBoxes']['logy'] = wx.CheckBox(self)
		self.plotOptions['sizer'].Add(self.plotOptions['checkBoxes']['logy'])
		self.Bind(wx.EVT_CHECKBOX, self.onPlotOptionsCheckbox, self.plotOptions['checkBoxes']['logy'])
		self.plotOptions['sizer'].Add(wx.StaticText(self, label="log-y"), 0, wx.LEFT | wx.EXPAND)

		self.plotOptions['checkBoxes']['dashedIfNegative'] = wx.CheckBox(self)
		self.plotOptions['sizer'].Add(self.plotOptions['checkBoxes']['dashedIfNegative'])
		self.Bind(wx.EVT_CHECKBOX, self.onPlotOptionsCheckbox, self.plotOptions['checkBoxes']['dashedIfNegative'])
		self.plotOptions['sizer'].Add(wx.StaticText(self, label="dashed-if-negative"), 0, wx.LEFT | wx.EXPAND)


		self.toolbar = matplotlib.backends.backend_wx.NavigationToolbar2Wx(self.canvas)
		self.toolbar.Realize()
		self.toolbar.Show()
		self.mainSizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)

		self._statusBar = wx.StatusBar(self.canvas)
		self._statusBar.SetFieldsCount(1)
		self.SetStatusBar(self._statusBar)
		self.canvas.mpl_connect('motion_notify_event', self._updateStatusBar)



		self.Fit()


		self.axes = self.figure.add_subplot(1, 1, 1)


		self.plotSelectionFrame = PlotSelectionFrame(self, plotter)
		self.plotSelectionFrame.Show(True)

	def update(self):
		self.updateLegend()
		self.updateCanvas()

	def clear(self):
		self.axes.cla()

	def updateCanvas(self):
		self.canvas.draw()

	def updateLegend(self):
		#box = self.axes.get_position()
		#self.axes.set_position([box.x0, box.y0, box.width * 0.9, box.height])
		handlesAndLegends = self.axes.get_legend_handles_labels()
		if len(handlesAndLegends[0]) != 0:
			self.legend = self.axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))


	def onPlotOptionsCheckbox(self, event):
		if event.GetId() == self.plotOptions['checkBoxes']['logx'].GetId():
			if event.IsChecked():
				self.plotter.setXScale('log')
			else:
				self.plotter.setXScale('linear')
		elif event.GetId() == self.plotOptions['checkBoxes']['logy'].GetId():
			if event.IsChecked():
				self.plotter.setYScale('log')
			else:
				self.plotter.setYScale('linear')
		elif event.GetId() == self.plotOptions['checkBoxes']['dashedIfNegative'].GetId():
			if event.IsChecked():
				self.plotter.setPlotDashedIfNegative(True)
			else:
				self.plotter.setPlotDashedIfNegative(False)
		else:
			raise ValueError("Cannot handle event")

		self.update()

	def setLogXCheckbox(self, value):
		"""
		Sets the checkbox that signifies that the x-axis is logarithmic.

		@param[in] value True to activate the checkbox, false to deactivate.
		"""

		self.plotOptions['checkBoxes']['logx'].SetValue(value)

	def setLogYCheckbox(self, value):
		"""
		Sets the checkbox that signifies that the y-axis is logarithmic.

		@param[in] value True to activate the checkbox, false to deactivate.
		"""

		self.plotOptions['checkBoxes']['logy'].SetValue(value)

	def setDashedIfNegativeCheckbox(self, value):
		"""
		Sets the checkbox that controls the plot-dashed-if-negative feature.

		@param[in] value True to activate the checkbox, false to deactivate.
		"""

		self.plotOptions['checkBoxes']['dashedIfNegative'].SetValue(value)

	def _updateStatusBar(self, event):
		"""
		Updates the status bar after an event happened.

		@param[in] event The event to react to.
		"""

		if event.inaxes:
			x, y = event.xdata, event.ydata
			self._statusBar.SetStatusText('x = ' + str(x) + ', y = ' + str(y))
