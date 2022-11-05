from .PlotSelectionList import PlotSelectionList

import wx

class PlotSelectionFrame(wx.Frame):
	"""
	Window for the plot selector.
	"""

	def __init__(self, parent, plotter):
		sizeX = 800
		sizeY = 600
		wx.Frame.__init__(self, parent, title='Plot Selection', size=(sizeX, sizeY))

		self.sizer = wx.BoxSizer(wx.VERTICAL)
		self.SetSizer(self.sizer)

		self.list = PlotSelectionList(self, plotter)
		self.sizer.Add(self.list, 1, wx.LEFT | wx.TOP | wx.EXPAND)
