def plotAxes(axes):
	if not isinstance(axes, list):
		axes = [axes]

	import wx

	class CanvasFrame(wx.Frame):
		def __init__(self):
			wx.Frame.__init__(self, None, -1, 'CanvasFrame')

			from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg

			self._statusBar = wx.StatusBar(self)
			self._statusBar.SetFieldsCount(1)
			self.SetStatusBar(self._statusBar)

			self._mainSizer = wx.BoxSizer(wx.VERTICAL)

			self._canvases = []
			self._canvasSizer = wx.GridSizer(1)
			for ax in axes:
				canvas = FigureCanvasWxAgg(self, -1, ax.get_figure())
				canvas.mpl_connect('motion_notify_event', self._updateStatusBar)

				sizer = wx.BoxSizer(wx.VERTICAL)
				self._addToolbar(canvas, sizer)
				sizer.Add(canvas, 1, wx.LEFT | wx.TOP | wx.EXPAND)

				self._canvases.append(canvas)
				self._canvasSizer.Add(sizer, 1, wx.LEFT | wx.TOP | wx.EXPAND)


			self._mainSizer.Add(self._canvasSizer, 1, wx.LEFT | wx.TOP | wx.EXPAND)
			self.SetSizer(self._mainSizer)
			self.Fit()



		def _updateStatusBar(self, event):
			"""
			Updates the status bar after an event happened.

			@param[in] event The event to react to.
			"""

			if event.inaxes:
				x, y = event.xdata, event.ydata
				self._statusBar.SetStatusText('x = ' + str(x) + ', y = ' + str(y))

		def _addToolbar(self, canvas, sizer):
			from matplotlib.backends.backend_wx import NavigationToolbar2Wx
			toolbar = NavigationToolbar2Wx(canvas)
			toolbar.Realize()
			sizer.Add(toolbar, 0, wx.LEFT | wx.TOP | wx.EXPAND)
			toolbar.update()


	class App(wx.App):
		def OnInit(self):
			frame = CanvasFrame()
			frame.Show(True)
			return True


	app = App(0)
	app.MainLoop()
