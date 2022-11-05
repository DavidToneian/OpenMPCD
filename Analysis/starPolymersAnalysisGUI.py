#! /usr/bin/env python

from MPCDAnalysis.Run import Run
from MPCDAnalysis.StarPolymersAnalysis.Acylindricity import Acylindricity
from MPCDAnalysis.StarPolymersAnalysis.Asphericity import Asphericity
from MPCDAnalysis.StarPolymersAnalysis.MagneticClusterCount import MagneticClusterCount
from MPCDAnalysis.StarPolymersAnalysis.OrientationAngles import OrientationAngles
from MPCDAnalysis.StarPolymersAnalysis.PotentialEnergy import PotentialEnergy
from MPCDAnalysis.StarPolymersAnalysis.RadiusOfGyration import RadiusOfGyration
from MPCDAnalysis.StarPolymersAnalysis.RelativeShapeAnisotropy import RelativeShapeAnisotropy
from MPCDAnalysis.StarPolymersAnalysis.RotationFrequencyVector import RotationFrequencyVector

import sys

if len(sys.argv) != 2:
	print("Usage: " + sys.argv[0] + " RUNDIR")
	exit(0)

run = Run(sys.argv[1])

import wx

class CanvasFrame(wx.Frame):
	def __init__(self):
		wx.Frame.__init__(self, None, -1, 'CanvasFrame')

		from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg

		plots = []
		plots.append(PotentialEnergy(run))

		plots.append(Acylindricity(run))
		plots.append(Asphericity(run))
		plots.append(MagneticClusterCount(run))
		plots.append(OrientationAngles(run))
		plots.append(RadiusOfGyration(run))
		plots.append(RelativeShapeAnisotropy(run))
		plots.append(RotationFrequencyVector(run))

		self._statusBar = wx.StatusBar(self)
		self._statusBar.SetFieldsCount(1)
		self.SetStatusBar(self._statusBar)

		self._mainSizer = wx.BoxSizer(wx.VERTICAL)

		self._canvases = []
		self._canvasSizer = wx.GridSizer(3)
		for plot in plots:
			axes = plot.getMPLAxesForValueAsFunctionOfTime()
			canvas = FigureCanvasWxAgg(self, -1, axes.get_figure())
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
exit(0)
