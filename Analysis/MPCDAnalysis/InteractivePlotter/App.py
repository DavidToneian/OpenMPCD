from .CanvasFrame import CanvasFrame

import wx

class App(wx.App):
	"""
	wxPython application class.
	"""

	def __init__(self, plotter):
		self.plotter = plotter
		wx.App.__init__(self)

	def OnInit(self):
		self.canvasFrame = CanvasFrame(self.plotter)
		self.SetTopWindow(self.canvasFrame)
		self.canvasFrame.Maximize(True)
		self.canvasFrame.Show(True)
		return True
