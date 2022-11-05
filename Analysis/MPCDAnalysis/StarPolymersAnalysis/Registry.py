## The one and only instance of `Registry`.
registry = None

class Registry:
	"""
	Class that registers the various analysis tools for star polymers.
	"""

	def __init__(self):
		"""
		The constructor.

		This is called automatically by the encolsing file, and must not be
		called thereafter.
		"""
		if registry is not None:
			raise Exception("Cannot create more than one instance")

		from collections import OrderedDict
		self._registry = OrderedDict()


	def register(self, cls):
		"""
		Register the given class with the registry.

		@param[in] cls The class to register.
		"""

		import inspect

		if not inspect.isclass(cls):
			raise ValueError()

		if cls.__name__ in self._registry:
			raise ValueError("Already in registry: " + cls.__name__)

		self._registry[cls.__name__] = cls


	def getRegisteredClasses(self):
		"""
		Returns a dictionary of registered classes.
		"""

		return self._registry


registry = Registry()
