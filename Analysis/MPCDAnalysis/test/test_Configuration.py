from MPCDAnalysis.Configuration import Configuration

def test_various():
	config = Configuration("""
		str = "hello world";
		int = 123;
		float = -4.56;
		group =
		{
			str = "foobar";
			int = -789;
			float = -10.0;
			group =
			{
				str = "level3";
			};
		};

		list = (0, 1, 2, 3);

		listOfGroups =
		(
			{
				name = "group1";
			},
			{
				name = "group2";
				isGroup2 = true;
			}
		);
	""")

	assert config["str"] == "hello world"
	assert config["int"] == 123
	assert config["group.str"] == "foobar"
	assert config["group"]["float"] == -10.0
	assert config["listOfGroups"]["[1].isGroup2"] == True

	config["str"] = "newvalue"
	config["group.str"] = "barfoo"
	config["listOfGroups"]["[1].isGroup2"] = False
	config["newsetting"] = "hello"
	config["newgroup.setting"] = 777
	assert config["str"] == "newvalue"
	assert config["group"]["str"] == "barfoo"
	assert config["listOfGroups"]["[1]"]["isGroup2"] == False
	assert config["listOfGroups"]["[1].isGroup2"] == False
	assert config["newsetting"] == "hello"
	assert config["newgroup.setting"] == 777

	assert "str" in config
	assert "int" in config
	assert "group" in config
	assert "group.str" in config
	assert "float" in config["group"]
	assert "group.group" in config
	assert "group.group.str" in config
	assert "foobar" not in config
	assert "list" in config
	assert "list.[0]" in config
	assert "[0]" in config["list"]
	assert "[4]" not in config["list"]
	assert "list.[4]" not in config
	assert "listOfGroups.[0].name" in config
	assert "[1]" in config["listOfGroups"]
	assert "name" in config["listOfGroups.[0]"]
	assert "listOfGroups.[1].name" in config
	assert "listOfGroups.[2].name" not in config

	assert "newgroup2" not in config
	newgroup2 = config.createGroup("newgroup2")
	config.createGroup("newgroup2.asdf.foo")
	newgroup2.createGroup("sub")
	newgroup2.createGroup("sub")
	assert "newgroup2" in config
	assert "newgroup2.asdf.foo" in config
	assert "newgroup2.sub" in config
	assert "sub" in config["newgroup2"]
	assert "sub" in newgroup2

	del config["newgroup2.asdf.foo"]
	assert "newgroup2.asdf.foo" not in config
	del config["newgroup2"]
	assert "newgroup2" not in config


	import copy

	assert config.isEquivalent(config)
	copiedConfig = copy.deepcopy(config)
	assert config.isEquivalent(copiedConfig)
	assert copiedConfig.isEquivalent(config)
	del config["newgroup.setting"]
	assert not config.isEquivalent(copiedConfig)
	assert not copiedConfig.isEquivalent(config)

	copiedConfig = copy.deepcopy(config)
	assert config.isEquivalent(copiedConfig)
	assert copiedConfig.isEquivalent(config)
	del config["newgroup"]
	assert not config.isEquivalent(copiedConfig)
	assert not copiedConfig.isEquivalent(config)

	copiedConfig = copy.deepcopy(config)
	del config["newsetting"]
	assert not config.isEquivalent(copiedConfig)
	assert not copiedConfig.isEquivalent(config)


	config = Configuration("foo: [1, 2, 3]")
	assert "foo" in config
	assert "foo.[0]" in config
	assert "[0]" in config["foo"]
	assert config["foo.[0]"] == 1
	assert config["foo.[1]"] == 2
	assert config["foo.[2]"] == 3
	config["foo.[0]"] = 4
	assert config["foo.[0]"] == 4
	assert config["foo.[1]"] == 2
	assert config["foo.[2]"] == 3
	assert config.isEquivalent(config)


def test_canonicalizing_constructor__LeesEdwards():
	configString = \
		"""
		group =
		{
			float = -10.0;
			group =
			{
				str = "level3";
			};
		};

		boundaryConditions:
		{
			type = "Lees-Edwards"
			shearRate = 0.123
		}

		list = (0, 1, 2, 3);
		"""


	configNotCanonicalized = Configuration(configString, False)
	configCanonicalized = Configuration(configString)

	assert "boundaryConditions" in configNotCanonicalized
	assert "boundaryConditions" in configCanonicalized


	assert "boundaryConditions.type" in configNotCanonicalized
	assert configNotCanonicalized["boundaryConditions.type"] == "Lees-Edwards"
	assert not "boundaryConditions.type" in configCanonicalized

	assert "type" in configNotCanonicalized["boundaryConditions"]
	assert \
		configNotCanonicalized["boundaryConditions"]["type"] == "Lees-Edwards"
	assert not "type" in configCanonicalized["boundaryConditions"]


	assert "boundaryConditions.shearRate" in configNotCanonicalized
	assert configNotCanonicalized["boundaryConditions.shearRate"] == 0.123
	assert not "boundaryConditions.shearRate" in configCanonicalized

	assert "shearRate" in configNotCanonicalized["boundaryConditions"]
	assert configNotCanonicalized["boundaryConditions"]["shearRate"] == 0.123
	assert not "shearRate" in configCanonicalized["boundaryConditions"]


	assert not "boundaryConditions.LeesEdwards" in configNotCanonicalized
	assert "boundaryConditions.LeesEdwards" in configCanonicalized

	assert \
		not "boundaryConditions.LeesEdwards.shearRate" in configNotCanonicalized
	assert "boundaryConditions.LeesEdwards.shearRate" in configCanonicalized

	assert \
		configCanonicalized["boundaryConditions.LeesEdwards.shearRate"] == 0.123
	assert \
		configCanonicalized["boundaryConditions"]["LeesEdwards"]["shearRate"] \
			== 0.123


def test_canonicalizing_constructor__cellSubdivision():
	configString = \
		"""
		group =
		{
			float = -10.0;
			group =
			{
				str = "level3";
			};
		};

		instrumentation:
		{
			cellSubdivision:
			{
				x = 5
				y = 6
				z = 7
			}
		}

		list = (0, 1, 2, 3);
		"""


	configNotCanonicalized = Configuration(configString, False)
	configCanonicalized = Configuration(configString)

	assert "instrumentation" in configNotCanonicalized
	assert "instrumentation" in configCanonicalized


	assert "instrumentation.cellSubdivision" in configNotCanonicalized
	assert configNotCanonicalized["instrumentation.cellSubdivision.x"] == 5
	assert configNotCanonicalized["instrumentation.cellSubdivision.y"] == 6
	assert configNotCanonicalized["instrumentation.cellSubdivision.z"] == 7
	assert not "instrumentation.cellSubdivision" in configCanonicalized

	assert "cellSubdivision" in configNotCanonicalized["instrumentation"]
	assert configNotCanonicalized["instrumentation"]["cellSubdivision.x"] == 5
	assert configNotCanonicalized["instrumentation"]["cellSubdivision.y"] == 6
	assert configNotCanonicalized["instrumentation"]["cellSubdivision.z"] == 7
	assert not "cellSubdivision" in configCanonicalized["instrumentation"]


	assert not "instrumentation.flowProfile" in configNotCanonicalized
	assert not "instrumentation.densityProfile" in configNotCanonicalized
	assert "instrumentation.flowProfile" in configCanonicalized
	assert "instrumentation.flowProfile.cellSubdivision" in configCanonicalized
	assert "instrumentation.densityProfile" in configCanonicalized
	assert \
		"instrumentation.densityProfile.cellSubdivision" in configCanonicalized

	assert \
		configCanonicalized["instrumentation.flowProfile.cellSubdivision.x"] \
		== 5
	assert \
		configCanonicalized["instrumentation.flowProfile.cellSubdivision.y"] \
		== 6
	assert \
		configCanonicalized["instrumentation.flowProfile.cellSubdivision.z"] \
		== 7

	assert \
		configCanonicalized["instrumentation"]["densityProfile"]\
		["cellSubdivision"]["x"] \
			== 5
	assert \
		configCanonicalized["instrumentation"]["densityProfile"]\
		["cellSubdivision"]["y"] \
			== 6
	assert \
		configCanonicalized["instrumentation"]["densityProfile"]\
		["cellSubdivision"]["z"] \
			== 7




def test_isEquivalent():
	config = Configuration("""
		str = "hello world";
		int = 123;
		float = -4.56;
		group =
		{
			str = "foobar";
			int = -789;
			float = -10.0;
			group =
			{
				str = "level3";
			};
		};

		list = (0, 1, 2, 3);

		listOfGroups =
		(
			{
				name = "group1";
			},
			{
				name = "group2";
				isGroup2 = true;
			}
		);
	""")

	import pytest

	with pytest.raises(TypeError):
		config.isEquivalent("string")
	with pytest.raises(TypeError):
		config.isEquivalent([config])
	with pytest.raises(TypeError):
		config.isEquivalent(config, (lambda x: x))
	with pytest.raises(TypeError):
		config.isEquivalent(config, [lambda x: x, "hi"])

	config["newsetting"] = "hello"
	config["newgroup.setting"] = 777

	import copy

	assert config.isEquivalent(config)
	copiedConfig = copy.deepcopy(config)
	assert config.isEquivalent(copiedConfig)
	assert copiedConfig.isEquivalent(config)
	del config["newgroup.setting"]
	assert not config.isEquivalent(copiedConfig)
	assert not copiedConfig.isEquivalent(config)

	copiedConfig = copy.deepcopy(config)
	assert config.isEquivalent(copiedConfig)
	assert copiedConfig.isEquivalent(config)
	del config["newgroup"]
	assert not config.isEquivalent(copiedConfig)
	assert not copiedConfig.isEquivalent(config)

	copiedConfig = copy.deepcopy(config)
	del config["newsetting"]
	assert not config.isEquivalent(copiedConfig)
	assert not copiedConfig.isEquivalent(config)


	config = Configuration("foo: [1, 2, 3]")
	config["foo.[0]"] = 4
	assert config.isEquivalent(config)


	configString = """
		toplevel = "hello world"
		group: { asdf = ["hi", "whatever"] }
		arrayOfStrings: ["foo", "bar", "helium"]
		arrayOfInts: [1, 2, 3]
		listOfStrings: ("foo", "bar", "helium")
		listOfInts: (1, 2, 3)
		"""

	translateX = lambda s: s.replace("h", "X")
	translateY = lambda s: s.replace("w", "Y")
	configStringReplaced = translateX(translateY(configString))
	config = Configuration(configString)
	configReplaced = Configuration(configStringReplaced)

	assert not config.isEquivalent(configReplaced)
	assert not config.isEquivalent(configReplaced, [translateX])
	assert not config.isEquivalent(configReplaced, [translateY])
	assert config.isEquivalent(configReplaced, [translateX, translateY])



def test_isEquivalent_array_list_vs_group():
	l = Configuration("x = (5, 10)")
	a = Configuration("x = [5, 10]")
	g = Configuration("x = {f = 5; t = 10}")

	assert l.isEquivalent(l)
	assert l.isEquivalent(a)
	assert a.isEquivalent(a)
	assert a.isEquivalent(l)
	assert g.isEquivalent(g)

	assert not l.isEquivalent(g)
	assert not a.isEquivalent(g)
	assert not g.isEquivalent(l)
	assert not g.isEquivalent(a)

def test_isEquivalent_array_list_sizes():
	listLong = Configuration("x = (2, 3, 4)")
	listShort = Configuration("x = (2, 3)")

	arrayLong = Configuration("x = [2, 3, 4]")
	arrayShort = Configuration("x = [2, 3]")

	longs = [listLong, arrayLong]
	shorts = [listShort, arrayShort]

	for long in longs:
		for otherLong in longs:
			assert long.isEquivalent(otherLong)

	for short in shorts:
		for otherShort in shorts:
			assert short.isEquivalent(otherShort)

	for long in longs:
		for short in shorts:
			assert not long.isEquivalent(short)
			assert not short.isEquivalent(long)


def test_isEquivalent_array_vs_list():
	config1 = Configuration("x = (2, 3, 4)")
	config2 = Configuration("x = [2, 3, 4]")
	config3 = Configuration("something = 2")

	assert config1.isEquivalent(config1)
	assert config2.isEquivalent(config2)

	assert not config1.isEquivalent(config3)
	assert not config2.isEquivalent(config3)
	assert not config3.isEquivalent(config1)
	assert not config3.isEquivalent(config2)

	assert config1.isEquivalent(config2)
	assert config2.isEquivalent(config1)


def test_getAsFlatDictionary():
	config = Configuration("""
		str = "hello world";
		int = 123;
		float = -4.56;
		group =
		{
			str = "foobar";
			int = -789;
			float = -10.0;
			group =
			{
				str = "level3";
			};
		};

		list = (0, 1, 2, 3);
		array = [ -1.23, 4.56 ];

		listOfGroups =
		(
			{
				name = "group1";
			},
			{
				name = "group2";
				isGroup2 = true;
			},
			{
			}
		);

		emptyGroup = {}
		emptyArray = []
		emptyList = ()
	""")

	d = config.getAsFlatDictionary()

	assert isinstance(d, dict)

	expectedElementCount = 0

	assert "str" in d
	assert d["str"] == "hello world"
	expectedElementCount += 1

	assert "int" in d
	assert d["int"] == 123
	expectedElementCount += 1

	assert "float" in d
	assert d["float"] == -4.56
	expectedElementCount += 1

	assert "group.str" in d
	assert d["group.str"] == "foobar"
	expectedElementCount += 1

	assert "group.int" in d
	assert d["group.int"] == -789
	expectedElementCount += 1

	assert "group.float" in d
	assert d["group.float"] == -10.0
	expectedElementCount += 1

	assert "group.group.str" in d
	assert d["group.group.str"] == "level3"
	expectedElementCount += 1

	assert "list.[0]" in d
	assert d["list.[0]"] == 0
	expectedElementCount += 1

	assert "list.[1]" in d
	assert d["list.[1]"] == 1
	expectedElementCount += 1

	assert "list.[2]" in d
	assert d["list.[2]"] == 2
	expectedElementCount += 1

	assert "list.[3]" in d
	assert d["list.[3]"] == 3
	expectedElementCount += 1

	assert "array.[0]" in d
	assert d["array.[0]"] == -1.23
	expectedElementCount += 1

	assert "array.[1]" in d
	assert d["array.[1]"] == 4.56
	expectedElementCount += 1

	assert "listOfGroups.[0].name" in d
	assert d["listOfGroups.[0].name"] == "group1"
	expectedElementCount += 1

	assert "listOfGroups.[1].name" in d
	assert d["listOfGroups.[1].name"] == "group2"
	expectedElementCount += 1

	assert "listOfGroups.[1].name" in d
	assert d["listOfGroups.[1].isGroup2"] == True
	expectedElementCount += 1

	assert "listOfGroups.[2]" in d
	assert d["listOfGroups.[2]"] is None
	expectedElementCount += 1

	assert "emptyGroup" in d
	assert d["emptyGroup"] == None
	expectedElementCount += 1

	assert "emptyArray" in d
	assert d["emptyArray"] == None
	expectedElementCount += 1

	assert "emptyList" in d
	assert d["emptyList"] == None
	expectedElementCount += 1

	assert len(d) == expectedElementCount


def test_getDifferencesAsFlatDictionary_argumentType():
	config = Configuration("foo = 1")
	rasied = False
	try:
		config.getDifferencesAsFlatDictionary(3)
	except TypeError:
		raised = True
	assert raised

def test_getDifferencesAsFlatDictionary():
	config1 = Configuration("""
		str = "hello world";
		int = 1234;
		float = -4.56;
		group =
		{
			str = "foobar";
			int = -789;
			float = -10.0;
			group =
			{
				str = "level3";
			};
		};

		list = (0, 1, 2, 3);

		listOfGroups =
		(
			{
				name = "group1";
			},
			{
				name = "group2";
				isGroup2 = true;
			},
			{
			}
		);

		emptyGroup = {}
	""")

	config2 = Configuration("""
		str = "hello world";
		int = 123;
		otherFloat = -4.56;
		group =
		{
			str = "baz";
			int = -789;
			float = -10.0;
		};

		list = (0, 1, 3);

		listOfGroups =
		(
			{
			},
			{
				name = "group2";
				isGroup2 = true;
			},
			{
			}
		);
	""")

	assert len(config1.getDifferencesAsFlatDictionary(config1)) == 0
	assert len(config2.getDifferencesAsFlatDictionary(config2)) == 0

	d1 = config1.getDifferencesAsFlatDictionary(config2)
	d2 = config2.getDifferencesAsFlatDictionary(config1)

	expectedElementCount = 0

	assert "int" in d1
	assert "int" in d2
	assert d1["int"] == [[1234], [123]]
	assert d2["int"] == [[123], [1234]]
	expectedElementCount += 1

	assert "float" in d1
	assert "float" in d2
	assert d1["float"] == [[-4.56], []]
	assert d2["float"] == [[], [-4.56]]
	expectedElementCount += 1

	assert "otherFloat" in d1
	assert "otherFloat" in d2
	assert d1["otherFloat"] == [[], [-4.56]]
	assert d2["otherFloat"] == [[-4.56], []]
	expectedElementCount += 1

	assert "group.str" in d1
	assert "group.str" in d2
	assert d1["group.str"] == [["foobar"], ["baz"]]
	assert d2["group.str"] == [["baz"], ["foobar"]]
	expectedElementCount += 1

	assert "group.group.str" in d1
	assert "group.group.str" in d2
	assert d1["group.group.str"] == [["level3"], []]
	assert d2["group.group.str"] == [[], ["level3"]]
	expectedElementCount += 1

	assert "list.[2]" in d1
	assert "list.[2]" in d2
	assert d1["list.[2]"] == [[2], [3]]
	assert d2["list.[2]"] == [[3], [2]]
	expectedElementCount += 1

	assert "list.[3]" in d1
	assert "list.[3]" in d2
	assert d1["list.[3]"] == [[3], []]
	assert d2["list.[3]"] == [[], [3]]
	expectedElementCount += 1

	assert "listOfGroups.[0]" in d1
	assert "listOfGroups.[0]" in d2
	assert d1["listOfGroups.[0]"] == [[], [None]]
	assert d2["listOfGroups.[0]"] == [[None], []]
	expectedElementCount += 1

	assert "listOfGroups.[0].name" in d1
	assert "listOfGroups.[0].name" in d2
	assert d1["listOfGroups.[0].name"] == [["group1"], []]
	assert d2["listOfGroups.[0].name"] == [[], ["group1"]]
	expectedElementCount += 1

	assert "emptyGroup" in d1
	assert "emptyGroup" in d2
	assert d1["emptyGroup"] == [[None], []]
	assert d2["emptyGroup"] == [[], [None]]
	expectedElementCount += 1

	assert len(d1) == expectedElementCount
	assert len(d2) == expectedElementCount



def test_saveToParameterFile():
	import pytest
	import tempfile

	with tempfile.NamedTemporaryFile("w") as tmpf:
		path = tmpf.name

		config = Configuration("""
			str = "hello world";
			int = 1234;
			float = -4.56;
			group =
			{
				str = "foobar";
				int = -789;
				float = -10.0;
				group =
				{
					str = "level3";
				};
			};

			list = (0, 1, 2, 3);

			listOfGroups =
			(
				{
					name = "group1";
				},
				{
					name = "group2";
					isGroup2 = true;
				},
				{
				}
			);

			emptyGroup = {}
		""")


		with pytest.raises(TypeError):
			config.saveToParameterFile(0)
		with pytest.raises(TypeError):
			config.saveToParameterFile("/tmp/foo", [])


		listOfAdditionalParameters = []
		listOfAdditionalParameters.append({})
		listOfAdditionalParameters.append(
			{
				"another-param" : 1.234,
				"additional-param": "foo",
			})

		for additionalParameters in listOfAdditionalParameters:
			config.saveToParameterFile(path, additionalParameters)

			with open(path, "r") as f:
				lineCount = 0

				for line in f:
					lineCount += 1

					if lineCount == 1:
						headerLine = line
					else:
						valueLine = line

				assert lineCount == 2

				flatDict = config.getAsFlatDictionary()
				flatDict.update(additionalParameters)

				assert headerLine == "\t".join(sorted(flatDict)) + "\n"

				expectedValueLine = ""
				for key in sorted(flatDict):
					if expectedValueLine:
						expectedValueLine += "\t"

					value = flatDict[key]

					if isinstance(value, str):
						expectedValueLine += "{" + value + "}"
					elif value is None:
						expectedValueLine += "None"
					elif isinstance(value, bool):
						expectedValueLine += "true" if value else "false"
					else:
						expectedValueLine += str(value)

				assert valueLine == expectedValueLine

def test___getitem__():
	config = Configuration("""
		str = "hello world";
		int = 123;
		float = -4.56;
		group =
		{
			str = "foobar";
			int = -789;
			float = -10.0;
			group =
			{
				str = "level3";
			};
		};

		list = (0, 1, 2, 3);

		listOfGroups =
		(
			{
				name = "group1";
			},
			{
				name = "group2";
				isGroup2 = true;
			}
		);
	""")


	import pytest
	with pytest.raises(TypeError):
		config.__getitem__(0)
	with pytest.raises(TypeError):
		config.__getitem__(config)

	with pytest.raises(KeyError):
		config.__getitem__("malformed..key")
	with pytest.raises(KeyError):
		config.__getitem__("malformedKey.")
	with pytest.raises(KeyError):
		config.__getitem__(".malformedKey")
	with pytest.raises(KeyError):
		config.__getitem__("nonexistent")


	assert config.__getitem__("str") == "hello world"
	assert config.__getitem__("int") == 123
	assert config.__getitem__("group.str") == "foobar"
	assert config.__getitem__("group").__getitem__("float") == -10.0
	assert config.__getitem__("listOfGroups").__getitem__("[1].isGroup2") == True

	config["str"] = "newvalue"
	config["group.str"] = "barfoo"
	config["listOfGroups"]["[1].isGroup2"] = False
	config["newsetting"] = "hello"
	config["newgroup.setting"] = 777
	assert config.__getitem__("str") == "newvalue"
	assert config.__getitem__("group").__getitem__("str") == "barfoo"
	assert \
		config.__getitem__("listOfGroups"). \
		__getitem__("[1]").__getitem__("isGroup2") == False
	assert \
		config.__getitem__("listOfGroups").__getitem__("[1].isGroup2") == False
	assert config.__getitem__("newsetting") == "hello"
	assert config.__getitem__("newgroup.setting") == 777



	config = Configuration("foo: [1, 2, 3]")
	assert config.__getitem__("foo.[0]") == 1
	assert config.__getitem__("foo.[1]") == 2
	assert config.__getitem__("foo.[2]") == 3
	config["foo.[0]"] = 4
	assert config.__getitem__("foo.[0]") == 4
	assert config.__getitem__("foo.[1]") == 2
	assert config.__getitem__("foo.[2]") == 3


def test___getitem___arrayInGroup():
	configString = """
		group:
		{
			array = [1.0, 0.0, -3.5]
		}
		"""
	config = Configuration(configString)

	assert config["group.array.[0]"] == 1.0
	assert config["group.array.[1]"] == 0.0
	assert config["group.array.[2]"] == -3.5

	assert config["group"]["array.[0]"] == 1.0
	assert config["group"]["array.[1]"] == 0.0
	assert config["group"]["array.[2]"] == -3.5

	assert config["group"]["array"]["[0]"] == 1.0
	assert config["group"]["array"]["[1]"] == 0.0
	assert config["group"]["array"]["[2]"] == -3.5


def test___setitem__():
	config = Configuration("""
		str = "hello world";
		int = 123;
		float = -4.56;
		group =
		{
			str = "foobar";
			int = -789;
			float = -10.0;
			group =
			{
				str = "level3";
			};
		};

		list = (0, 1, 2, 3);

		listOfGroups =
		(
			{
				name = "group1";
			},
			{
				name = "group2";
				isGroup2 = true;
			}
		);
	""")


	import pytest
	with pytest.raises(TypeError):
		config.__setitem__(0, 0)
	with pytest.raises(TypeError):
		config.__setitem__(config, 0)

	with pytest.raises(KeyError):
		config.__setitem__("", 0)
	with pytest.raises(KeyError):
		config.__setitem__("malformed..key", 0)
	with pytest.raises(KeyError):
		config.__setitem__("malformedKey.", 0)
	with pytest.raises(KeyError):
		config.__setitem__(".malformedKey", 0)


	assert config["str"] == "hello world"
	assert config["int"] == 123
	assert config["group.str"] == "foobar"
	assert config["group"]["float"] == -10.0
	assert config["listOfGroups"]["[1].isGroup2"] == True

	config.__setitem__("str", "newvalue")
	config.__setitem__("group.str", "barfoo")
	config["listOfGroups"].__setitem__("[1].isGroup2", False)
	config.__setitem__("newsetting", "hello")
	config.__setitem__("newgroup.setting", 777)
	assert config["str"] == "newvalue"
	assert config["group"]["str"] == "barfoo"
	assert config["listOfGroups"]["[1]"]["isGroup2"] == False
	assert config["listOfGroups"]["[1].isGroup2"] == False
	assert config["newsetting"] == "hello"
	assert config["newgroup.setting"] == 777

	config = Configuration("foo: [1, 2, 3]")
	assert config["foo.[0]"] == 1
	assert config["foo.[1]"] == 2
	assert config["foo.[2]"] == 3
	config.__setitem__("foo.[0]", 4)
	assert config["foo.[0]"] == 4
	assert config["foo.[1]"] == 2
	assert config["foo.[2]"] == 3


def test___setitem___newValueIsGroup():
	configString = """
		topSetting = 1
		group:
		{
			array = [1.0, 0.0, -3.5]
			subgroup:
			{
				value = "string"
			}
		}
		"""
	config = Configuration(configString)

	config["copy"] = config["group"]

	assert config["copy.array.[0]"] == 1.0
	assert config["copy.array.[1]"] == 0.0
	assert config["copy.array.[2]"] == -3.5
	assert config["copy.subgroup.value"] == "string"


	config["newGroup.copy"] = config["group"]

	assert config["newGroup.copy.array.[0]"] == 1.0
	assert config["newGroup.copy.array.[1]"] == 0.0
	assert config["newGroup.copy.array.[2]"] == -3.5
	assert config["newGroup.copy.subgroup.value"] == "string"


def test___delitem__():
	configString = """
		topSetting = 1
		group:
		{
			array = [1.0, 0.0, -3.5]
			subgroup:
			{
				value = "string"
			}
		}
		"""
	config = Configuration(configString)

	import pytest
	with pytest.raises(TypeError):
		del config[0]

	assert "topSetting" in config
	assert "group" in config
	assert "group.array" in config
	assert "group.subgroup" in config
	assert "group.subgroup.value" in config


	del config["foo"]
	assert "topSetting" in config
	assert "group" in config
	assert "group.array" in config
	assert "group.subgroup" in config
	assert "group.subgroup.value" in config

	del config["group.foo"]
	assert "topSetting" in config
	assert "group" in config
	assert "group.array" in config
	assert "group.subgroup" in config
	assert "group.subgroup.value" in config

	del config["foo.bar"]
	assert "topSetting" in config
	assert "group" in config
	assert "group.array" in config
	assert "group.subgroup" in config
	assert "group.subgroup.value" in config


	del config["group.array"]
	assert "topSetting" in config
	assert "group" in config
	assert not "group.array" in config
	assert "group.subgroup" in config
	assert "group.subgroup.value" in config

	del config["topSetting"]
	assert not "topSetting" in config
	assert "group" in config
	assert not "group.array" in config
	assert "group.subgroup" in config
	assert "group.subgroup.value" in config

	del config["group"]
	assert not "topSetting" in config
	assert not "group" in config
	assert not "group.array" in config
	assert not "group.subgroup" in config
	assert not "group.subgroup.value" in config


def test___contains__():
	configString = """
		topSetting = 1
		group:
		{
			array = [1.0, 0.0, -3.5]
			subgroup:
			{
				value = "string"
			}
		}
		"""
	config = Configuration(configString)

	import pytest
	with pytest.raises(TypeError):
		del config[0]

	assert "topSetting" in config
	assert "group" in config
	assert "group.array" in config
	assert "group.subgroup" in config
	assert "group.subgroup.value" in config

	assert not "foo" in config
	assert not "group.array.subgroup" in config
	assert not "group.array.subgroup.value" in config
	assert not "group.subgroup.foo" in config
	assert not "group.foo" in config
	assert not "group.foo.bar" in config


def test___repr__():
	configString = """
		topSetting = 1
		group:
		{
			array = [1.0, 0.0, -3.5]
			subgroup:
			{
				value = "string"
			}
		}
		"""

	config = Configuration("")
	assert isinstance(config.__repr__(), str)

	config = Configuration(configString)
	assert isinstance(config.__repr__(), str)

	loadedConfig = Configuration(config.__repr__())
	assert loadedConfig.isEquivalent(config)
