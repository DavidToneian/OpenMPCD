class Configuration:
	"""
	Represents an OpenMPCD configuration.

	Configurations are assumed to be in the libconfig++ format.
	"""

	def __init__(self, source = None, canonicalize = True):
		"""
		The constructor.

		@param[in] source
		           If `None`, constructs an empty configuration.
		           If the parameter is a string that corresponds to a file, it
		           is loaded as the configuration file.
		           If the string corresponds to a directory, it is assumed that
		           this directory contains a file `config.txt`, which is loaded.
		           Otherwise, it is assumed that the parameter contains a string
		           representation of a configuration.
		@param[in] canonicalize
		           Set to true to canonicalize the input configuration, i.e.
		           transform it such that deprecated configuration settings are
		           translated into their new equivalents.
		"""

		import os.path
		import pylibconfig2

		self._config = pylibconfig2.Config()

		if source is not None:
			if os.path.isfile(source):
				self._config.read_file(source)
			elif os.path.isdir(source):
				self._config.read_file(source + "/config.txt")
			else:
				self._config.read_string(source)

			if canonicalize:
				self._canonicalize()


	def createGroup(self, name):
		"""
		Creates a group of the given name, if it does not exist, and returns it.

		This will create groups recursively, if necessary.
		"""

		import pylibconfig2

		if self.__contains__(name):
			ret = self._config.get(name)
			if not isinstance(ret, pylibconfig2.ConfGroup):
				raise Exception()

			return ret

		components = name.split(".")
		current = self._config
		for component in components:
			tmp = Configuration()
			tmp._config = current
			if component not in tmp:
				current.set(component, pylibconfig2.ConfGroup())
			current = current.get(component)

		if not isinstance(current, pylibconfig2.ConfGroup):
			raise Exception

		return self.__getitem__(name)


	def isEquivalent(self, rhs, pathTranslators = []):
		"""
		Returns whether this configuration is equivalent to `rhs`, i.e. if all
		settings in this configuration exist and have the same values in `rhs`
		and vice versa.

		This method does not discriminate what `libconfig` calls arrays and
		lists.

		@throw TypeError
		       Throws if `rhs` is not of type `Configuration`.
		@throw TypeError
		       Throws if `pathTranslators` is not a list of elements that
		       `callable()` returns `True` for.

		@param[in] rhs
		           The instance to compare to.
		@param[in] pathTranslators
		           Any configuration value string (on both this instance and
		           `rhs`) will be successively replaced by the results of
		           calling the functions in `pathTranslators` on that string.
		"""

		def applyTanslators(s, translators):
			for translator in translators:
				s = translator(s)
			return s

		if not isinstance(rhs, Configuration):
			raise TypeError()
		if not isinstance(pathTranslators, list):
			raise TypeError()
		for translator in pathTranslators:
			if not callable(translator):
				raise TypeError()

		import pylibconfig2
		listTypes = (pylibconfig2.ConfList, pylibconfig2.ConfArray)
		if isinstance(self._config, listTypes):
			if not isinstance(rhs._config, listTypes):
				return False
			if len(self._config) != len(rhs._config):
				return False

			for i in range(0, len(self._config)):
				key = "[" + str(i) + "]"
				if key not in rhs:
					return False
				if isinstance(self[key], self.__class__):
					if not isinstance(rhs[key], self.__class__):
						return False
					if not self[key].isEquivalent(rhs[key], pathTranslators):
						return False
				else:
					lhsValue = self[key]
					rhsValue = rhs[key]

					if type(lhsValue) != type(rhsValue):
						return False

					if isinstance(lhsValue, str):
						lhsString = applyTanslators(lhsValue, pathTranslators)
						rhsString = applyTanslators(rhsValue, pathTranslators)
						if lhsString != rhsString:
							return False
					else:
						if lhsValue != rhsValue:
							return False

			return True

		if isinstance(rhs._config, listTypes):
			return False

		for key in self._config.__dict__:
			if key not in rhs:
				return False

			if isinstance(self[key], Configuration):
				if not isinstance(rhs[key], Configuration):
					return False
				if not self[key].isEquivalent(rhs[key], pathTranslators):
					return False
			elif isinstance(self[key], str):
				if not isinstance(rhs[key], str):
					return False
				lhsString = applyTanslators(self[key], pathTranslators)
				rhsString = applyTanslators(rhs[key], pathTranslators)
				if lhsString != rhsString:
					return False
			else:
				if self[key] != rhs[key]:
					return False

		for key in rhs._config.__dict__:
			if key not in self:
				return False

		return True


	def get(self, key):
		"""
		Alias for `__getitem__`, which makes the interface more compatible to
		`pylibconfig2.Config`.
		"""

		return self[key]


	def getAsFlatDictionary(self):
		"""
		Returns this configuration as a dictionary, with each key being a
		setting name (including the names of the parent setting groups, lists,
		etc.), and the corresponding dictionary value being the setting's value.

		Empty groups, lists, and arrays will have `None` as their value.
		"""

		import pylibconfig2

		if isinstance(self._config, pylibconfig2.ConfArray):
			ret = {}
			for key, value in enumerate(self._config):
				ret["[" + str(key) + "]"] = value
			return ret

		ret = {}
		for key in self._config.__dict__:
			if isinstance(self[key], Configuration):
				subdict = self[key].getAsFlatDictionary()
				for subkey, subvalue in subdict.items():
					ret[key + "." + subkey] = subvalue
				if len(subdict) == 0:
					ret[key] = None
			else:
				ret[key] = self[key]

		if isinstance(self._config, pylibconfig2.ConfList):
			for i in range(0, len(self._config)):
				key = "[" + str(i) + "]"
				if isinstance(self[key], Configuration):
					subdict = self[key].getAsFlatDictionary()
					for subkey, subvalue in subdict.items():
						ret[key + "." + subkey] = subvalue
					if len(subdict) == 0:
						ret[key] = None
				else:
					ret[key] = self[key]

		return ret


	def getDifferencesAsFlatDictionary(self, rhs):
		"""
		Returns a dictionary, the keys of which are exactly the setting names
		that appear in either exactly one of `self` and `rhs`, or occur in both,
		but with different values.

		Each key's value is a list of two lists; the first corresponding to
		`self`, the second to `rhs`. These lists are empty if the setting is not
		set in the corresponding configuration, and otherwise contain the value
		that is set in the corresponding configuration.

		Setting names and values are understood as in `getAsFlatDictionary`.

		@param[in] rhs
		           The right-hand-side instance of this class.
		"""

		if not isinstance(rhs, self.__class__):
			raise TypeError()

		myDict = self.getAsFlatDictionary()
		rhsDict = rhs.getAsFlatDictionary()

		ret = {}
		for myKey in myDict:
			if myKey not in rhsDict:
				ret[myKey] = [[myDict[myKey]], []]
			elif myDict[myKey] != rhsDict[myKey]:
				ret[myKey] = [[myDict[myKey]], [rhsDict[myKey]]]

		for rhsKey in rhsDict:
			if rhsKey not in myDict:
				ret[rhsKey] = [[], [rhsDict[rhsKey]]]


		return ret


	def saveToParameterFile(self, path, additionalParameters = {}):
		"""
		Saves the configuration settings in a file with the given `path`.

		The first line contains the setting names, ordered lexicographically and
		separated by tab characters;
		the second line contains the corresponding values, with string values
		enclosed in curly braces, empty groups being represented by the string
		(not enclosed in curly braces) `None`,
		and with boolean values being represented by the strings `true` and
		`false`.

		This format is supposed to be readable by the `LaTeX` package
		`pgfplotstable`.

		@throw TypeError
		       Throws if any argument has an invalid type.

		@param[in] path
		           The file path to save the output to, as a string.
		@param[in] additionalParameters
		           A dictionary containing, the keys of which are strings that
		           describe addition parameters to write to the file, while the
		           repsective values describe the parameter values.
		"""

		if not isinstance(path, str):
			raise TypeError()

		if not isinstance(additionalParameters, dict):
			raise TypeError()
		for key in additionalParameters:
			if not isinstance(key, str):
				raise TypeError()


		flatDict = self.getAsFlatDictionary()
		for key, value in additionalParameters.items():
			assert key not in flatDict
			flatDict[key] = value

		headerLine = ""
		valueLine = ""
		for key in sorted(flatDict):
			if headerLine:
				headerLine += "\t"
				valueLine += "\t"

			headerLine += key

			value = flatDict[key]
			if isinstance(value, str):
				valueLine += "{" + value + "}"
			elif value is None:
				valueLine += "None"
			elif isinstance(value, bool):
				if value:
					valueLine += "true"
				else:
					valueLine += "false"
			else:
				valueLine += str(value)

		with open(path, "w") as f:
			f.write(headerLine + "\n" + valueLine)


	def __getitem__(self, key):
		"""
		Returns the value of the given key.

		@throw TypeError
		       Throws if `key` is not of type `str`.
		@throw KeyError
		       Throws if `key` is malformed or does not exist.

		@param[in] key
		           The key to retrieve.
		"""

		import pylibconfig2

		if not isinstance(key, str):
			raise TypeError

		listTypes = (pylibconfig2.ConfList, pylibconfig2.ConfArray)
		pylibconfigBaseTypes = listTypes + (pylibconfig2.ConfGroup,)

		if isinstance(self._config, pylibconfig2.Config):
			ret = self._config.lookup(key)
		elif isinstance(self._config, pylibconfig2.ConfGroup):
			components = key.split(".")
			ret = self._config
			for i, component in enumerate(components):
				if component not in ret.keys():
					raise KeyError(key)
				ret = ret.get(component)
				if isinstance(ret, listTypes):
					tmp = Configuration()
					tmp._config = ret
					if i + 1 == len(components):
						return tmp

					newKey = ".".join(components[i + 1:])
					return tmp[newKey]
		elif isinstance(self._config, listTypes):
			components = key.split(".", 1)
			numeral = components[0]
			if numeral[0] != "[" or numeral[-1] != "]":
				raise KeyError(key)
			number = int(numeral[1:-1])
			if str(number) != numeral[1:-1]:
				raise KeyError(key)

			ret = self._config[number]

			if isinstance(ret, pylibconfigBaseTypes):
				tmp = Configuration()
				tmp._config = ret

				if len(components) == 1:
					return tmp

				return tmp[".".join(components[1:])]

			return ret
		else:
			raise Exception

		if ret is None:
			raise KeyError(key)

		if isinstance(ret, pylibconfigBaseTypes):
			ret2 = Configuration()
			ret2._config = ret
			return ret2

		return ret


	def __setitem__(self, key, value):
		"""
		Sets the given configuration key to the given value.

		Groups will be created as needed, if they do not exist already.

		@throw TypeError
		       Throws if `key` is not of type `str`.
		@throw KeyError
		       Throws if `key` is malformed.

		@param[in] key
		           The configuration key.
		@param[in] value
		           The configuration value.
		"""

		if not isinstance(key, str):
			raise TypeError()
		if len(key) == 0:
			raise KeyError()

		components = key.split(".")

		for component in components:
			if len(component) == 0:
				raise KeyError()

		if isinstance(value, Configuration):
			value = value._config

		if len(components) == 1:
			self._config.set(key, value)
		else:
			group = self
			for component in components[:-1]:
				if not component in group:
					if component[0] == "[":
						if not component[-1] == "]":
							raise KeyError(key)
						raise Exception("not implemented")
					group.createGroup(component)
				group = group[component]

			import pylibconfig2
			if isinstance(group._config, pylibconfig2.ConfArray):
				component = components[-1]
				if component[0] != "[" or component[-1] != "]":
					raise KeyError(key)
				number = int(component[1:-1])
				group._config[number] = value
			else:
				group._config.set(components[-1], value)


	def __delitem__(self, key):
		"""
		Deletes the given setting recursively, if it exists.

		@throw TypeError
		       Throws if `key` is not an instance of `str`.

		@param[in] key
		           The setting to delete, as a string.
		"""

		if not isinstance(key, str):
			raise TypeError()

		if key not in self:
			return

		components = key.split(".")

		if len(components) == 1:
			del self._config.__dict__[key]
			return

		parent = self[".".join(components[:-1])]
		del parent[components[-1]]


	def __contains__(self, key):
		"""
		Returns whether the given configuration `key` has been set.

		@throw TypeError
		       Throws if `key` is not an instance of `str`.

		@param[in] key
		           The setting to look up, as a string.
		"""

		import pylibconfig2

		if not isinstance(key, str):
			raise KeyError

		if isinstance(self._config, pylibconfig2.Config):
			try:
				return self._config.lookup(key) is not None
			except pylibconfig2.ParseException:
				return False

		if isinstance(self._config, pylibconfig2.ConfGroup):
			components = key.split(".")
			ret = self._config
			for component in components:
				if component not in ret.keys():
					return False
				ret = ret.get(component)

			return True

		listTypes = (pylibconfig2.ConfList, pylibconfig2.ConfArray)
		if isinstance(self._config, listTypes):
			components = key.split(".", 1)
			numeral = components[0]
			if numeral[0] != "[" or numeral[-1] != "]":
				raise KeyError(key)
			number = int(numeral[1:-1])
			if str(number) != numeral[1:-1]:
				raise KeyError(key)

			try:
				self._config[number]
			except IndexError:
				return False
			return True

		raise Exception()


	def __repr__(self):
		"""
		Returns a `str` representation of this instance.

		This representation can be used to construct another instance of this
		class that is equivalent to this instance.
		"""

		return self._config.__repr__()


	def _canonicalize(self):
		"""
		Transforms this instance's configuration data such that deprecated
		settings are translated into their replacements.
		"""

		if "boundaryConditions.type" in self:
			if self["boundaryConditions.type"] == "Lees-Edwards":
				shearRate = self["boundaryConditions.shearRate"]
				del self["boundaryConditions.type"]
				del self["boundaryConditions.shearRate"]
				self["boundaryConditions.LeesEdwards.shearRate"] = shearRate
			else:
				raise RuntimeError("Unknown boundary condition")

		if "instrumentation.cellSubdivision" in self:
			assert "instrumentation.flowProfile" not in self
			assert "instrumentation.densityProfile" not in self

			cellSubdivision = self["instrumentation.cellSubdivision"]
			del self["instrumentation.cellSubdivision"]

			self["instrumentation.flowProfile.cellSubdivision"] = \
				cellSubdivision
			self["instrumentation.densityProfile.cellSubdivision"] = \
				cellSubdivision
