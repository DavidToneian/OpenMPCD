import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy

def findRangesOfEqualSign(data):
	if len(data) == 0:
		return []

	sign = None
	lastPos = None
	ranges = []
	for index, value in enumerate(data):
		pos = value >= 0
		if sign is None:
			sign = pos
			lastPos = 0
			continue

		if pos != sign:
			range_ = [lastPos, index]
			ranges.append((range_, sign))

			lastPos = index
			sign = not sign

	range_ = [lastPos, len(data)]
	ranges.append((range_, sign))

	return ranges

def plotDashedIfNegative(x, y, yErrors=None, logY=False, defaultLowerErrorBound=1e-30, ** kwargs):
	ax = kwargs.pop('ax', plt.gca())
	kwargs.pop('linestyle', None)

	ranges = findRangesOfEqualSign(y)

	if not ranges:
		return ax.plot(x, y, **kwargs)

	color = None
	lines = []
	for range_, sign in ranges:
		linestyle = '-' if sign else '--'

		currentX = x[range_[0]:range_[1]]
		currentY = y[range_[0]:range_[1]]
		if not sign:
			currentY = -numpy.array(currentY)
		current_line, = ax.plot(currentX, currentY, linestyle=linestyle, **kwargs)
		lines.append(current_line)

		if color is None:
			color = current_line.get_color()
			kwargs['color'] = color
			kwargs.pop('label', None)

	if yErrors is not None:
		lowerBounds = []
		upperBounds = []

		for key, value in enumerate(y):
			lower = value - yErrors[key]
			upper = value + yErrors[key]

			if value < 0:
				lower, upper = -upper, -lower

			if logY:
				if lower < 0:
					lower = defaultLowerErrorBound

				if upper < 0:
					upper = defaultLowerErrorBound

			lowerBounds.append(lower)
			upperBounds.append(upper)

		ax.fill_between(x, lowerBounds, upperBounds, alpha=0.3)

	return lines


class DiscreteSliderWidget(matplotlib.widgets.Slider):
	"""
	A matplotlib-compatible slider widget, the values of which are discrete.
	"""

	def __init__(self, *args, **kwargs):
		"""
		The constructor.

		The following keyword arguments are accepted for speacial treatment in
		this class:
		- `stepSize` specifies the step size with which the slider is allowed
		  to move. That is, values that can be taken on by the slider are
		  integer multiples of `stepSize`. This value must be of type `int` and
		  be positive. Defaults to `1`.

		@throw TypeError
		       Throws if any of the parameters described in the main body of
		       the documentation of this class is of the wrong type.
		@throw ValueError
		       Throws if any of the parameters described in the main body of
		       the documentation of this class has an invalid value.

		@param[in] args
		           Positional arguments, which will be passed to the base class'
		           constructor.
		@param[in] kwargs
		           Keyword arguments, which will be passed to the base class'
		           constructor, with the exception of the keyword arguments
		           described in the main body of the documentation of this
		           function.
		"""

		self._stepSize = kwargs.pop('stepSize', 1)
		self._valinit = kwargs.get('valinit', None)

		if isinstance(self._valinit, (int, float)) and self._valinit == 0:
			#workaround for broken rendering of slider
			kwargs['valinit'] = 1e-10

		if not isinstance(self._stepSize, int):
			raise TypeError()
		if self._stepSize <= 0:
			raise ValueError()

		matplotlib.widgets.Slider.__init__(self, *args, **kwargs)


	def set_val(self, continuousValue):
		"""
		Called when the slider moves.

		@param[in] continuousValue
		           The value the slider has been moved to, as a continuous
		           variable.
		"""

		self.val = continuousValue

		discreteValue = int(continuousValue // self._stepSize) * self._stepSize

		self.poly.xy[2][0] = discreteValue
		self.poly.xy[3][0] = discreteValue

		self.valtext.set_text(self.valfmt % discreteValue)

		if self.drawon:
			self.ax.figure.canvas.draw()

		if self.eventson:
			for callback in self.observers.values():
				callback(discreteValue)
