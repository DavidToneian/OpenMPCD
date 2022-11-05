from .. import Exceptions

import wx.lib.mixins.listctrl

# see http://code.activestate.com/recipes/426407-columnsortermixin-with-a-virtual-wxlistctrl/

class PlotSelectionList(wx.ListCtrl, wx.lib.mixins.listctrl.ListCtrlAutoWidthMixin, wx.lib.mixins.listctrl.ColumnSorterMixin):
	"""
	List of plots and their characteristics, which can be used to select which plots to draw.
	"""

	def __init__(self, parent, plotter):
		wx.ListCtrl.__init__(self, parent, style=wx.LC_REPORT | wx.LC_VIRTUAL | wx.LC_HRULES | wx.LC_VRULES)

		self.plotter = plotter
		self.plotter._registerPlotSelector(self)

		self.images = {}
		self.imagelist = wx.ImageList(16, 16)
		self.images['sort_down'] = self.imagelist.Add(wx.ArtProvider.GetBitmap(wx.ART_GO_DOWN, wx.ART_TOOLBAR, (16, 16)))
		self.images['sort_up'] = self.imagelist.Add(wx.ArtProvider.GetBitmap(wx.ART_GO_UP, wx.ART_TOOLBAR, (16, 16)))
		self.images['tick_mark'] = self.imagelist.Add(wx.ArtProvider.GetBitmap(wx.ART_TICK_MARK, wx.ART_TOOLBAR, (16, 16)))
		self.images['cross_mark'] = self.imagelist.Add(wx.ArtProvider.GetBitmap(wx.ART_CROSS_MARK, wx.ART_TOOLBAR, (16, 16)))
		self.SetImageList(self.imagelist, wx.IMAGE_LIST_SMALL)

		wx.lib.mixins.listctrl.ListCtrlAutoWidthMixin.__init__(self)
		wx.lib.mixins.listctrl.ColumnSorterMixin.__init__(self, 0)

		self.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.OnItemActivated)

	def getColumnName(self, column):
		"""
		Returns a column'a name.

		@param[in] column The column ID, from the range [0, GetColumnCount()).
		@throw OutOfRangeException Throws if column is out of range.
		"""

		if column >= self.GetColumnCount():
			raise Exceptions.OutOfRangeException("'column' out of range.")

		return self.GetColumn(column).GetText()

	def getItemIndex(self, item):
		return self.itemIndexMap[item]

	def getItemData(self, item):
		return self.itemDataMap[self.getItemIndex(item)]

	def getItemColumn(self, item, column):
		itemData = self.getItemData(item)
		return itemData[column]

	def GetListCtrl(self):
		return self

	def GetSortImages(self):
		return (self.images['sort_down'], self.images['sort_up'])

	def OnGetItemText(self, item, col):
		return self.getItemColumn(item, col)

	def OnGetItemImage(self, item):
		if self.getItemColumn(item, 0):
			return self.images['tick_mark']

		return self.images['cross_mark']

	def OnGetItemAttr(self, item):
		return None

	def SortItems(self, sorter):
		items = list(self.itemDataMap.keys())
		items.sort(sorter)
		self.itemIndexMap = items
		self.Refresh()

	def OnItemActivated(self, event):
		self.plotter.toggleVisibility(self.itemIndexMap[event.m_itemIndex])
