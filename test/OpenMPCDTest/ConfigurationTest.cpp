/**
 * @file
 * Tests `OpenMPCD::Configuration`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/Configuration.hpp>

#include <boost/filesystem/operations.hpp>

SCENARIO(
	"`OpenMPCD::Configuration::Setting::hasName`"
	"")
{
	using OpenMPCD::Configuration;

	libconfig::Config config;
	config.readFile("test/data/Configuration/Setting.txt");

	GIVEN("the root setting")
	{
		const Configuration::Setting s(config.getRoot());
		THEN("`hasName() == false`")
		{
			REQUIRE_FALSE(s.hasName());
		}
	}

	GIVEN("a setting of type 'string'")
	{
		const Configuration::Setting s(config.lookup("str"));
		THEN("`hasName() == true`")
		{
			REQUIRE(s.hasName());
		}
	}

	GIVEN("a setting of integer type")
	{
		const Configuration::Setting s(config.lookup("int"));
		THEN("`hasName() == true`")
		{
			REQUIRE(s.hasName());
		}
	}

	GIVEN("a setting of float type")
	{
		const Configuration::Setting s(config.lookup("float"));
		THEN("`hasName() == true`")
		{
			REQUIRE(s.hasName());
		}
	}

	GIVEN("a setting group")
	{
		const Configuration::Setting s(config.lookup("group"));
		THEN("`hasName() == true`")
		{
			REQUIRE(s.hasName());
		}
	}

	GIVEN("a setting list")
	{
		const Configuration::Setting s(config.lookup("list"));
		THEN("`hasName() == true`")
		{
			REQUIRE(s.hasName());
		}
	}

	GIVEN("an entry in a setting list")
	{
		const Configuration::Setting s(config.lookup("list")[0]);
		THEN("`hasName() == false`")
		{
			REQUIRE_FALSE(s.hasName());
		}
	}
}


SCENARIO(
	"`OpenMPCD::Configuration::Setting::getName`"
	"")
{
	using OpenMPCD::Configuration;

	libconfig::Config config;
	config.readFile("test/data/Configuration/Setting.txt");

	#ifdef OPENMPCD_DEBUG
		GIVEN("the root setting")
		{
			const Configuration::Setting s(config.getRoot());
			THEN("`getName()` throws")
			{
				REQUIRE_THROWS_AS(
					s.getName(),
					OpenMPCD::InvalidCallException);
			}
		}
	#endif

	GIVEN("a setting of type 'string'")
	{
		const Configuration::Setting s(config.lookup("str"));
		THEN("`getName()` works")
		{
			REQUIRE(s.getName() == "str");
		}
	}

	GIVEN("a setting of integer type")
	{
		const Configuration::Setting s(config.lookup("int"));
		THEN("`getName()` works")
		{
			REQUIRE(s.getName() == "int");
		}
	}

	GIVEN("a setting of float type")
	{
		const Configuration::Setting s(config.lookup("float"));
		THEN("`getName()` works")
		{
			REQUIRE(s.getName() == "float");
		}
	}

	GIVEN("a setting group")
	{
		const Configuration::Setting s(config.lookup("group"));
		THEN("`getName()` works")
		{
			REQUIRE(s.getName() == "group");
		}
	}

	GIVEN("a setting list")
	{
		const Configuration::Setting s(config.lookup("list"));
		THEN("`getName()` works")
		{
			REQUIRE(s.getName() == "list");
		}
	}

	#ifdef OPENMPCD_DEBUG
		GIVEN("an entry in a setting list")
		{
			const Configuration::Setting s(config.lookup("list")[0]);
			THEN("`getName()` throws")
			{
				REQUIRE_THROWS_AS(
					s.getName(),
					OpenMPCD::InvalidCallException);
			}
		}
	#endif
}


SCENARIO(
	"`OpenMPCD::Configuration::Setting::has`"
	"")
{
	using OpenMPCD::Configuration;

	libconfig::Config config;
	config.readFile("test/data/Configuration/Setting.txt");

	GIVEN("the root setting")
	{
		const Configuration::Setting s(config.getRoot());

		THEN("`has` throws if given an illegal setting name")
		{
			REQUIRE_THROWS_AS(
				s.has(".float"),
				OpenMPCD::InvalidArgumentException);
			REQUIRE_THROWS_AS(
				s.has("float."),
				OpenMPCD::InvalidArgumentException);
			REQUIRE_THROWS_AS(
				s.has("."),
				OpenMPCD::InvalidArgumentException);
		}

		THEN("`has` works")
		{
			REQUIRE(s.has("str"));
			REQUIRE(s.has("int"));
			REQUIRE(s.has("float"));

			REQUIRE(s.has("group"));
			REQUIRE(s.has("group.str"));
			REQUIRE(s.has("group.int"));
			REQUIRE(s.has("group.float"));
			REQUIRE(s.has("group.group"));
			REQUIRE(s.has("group.group.str"));

			REQUIRE(s.has("list"));

			REQUIRE(s.has("listOfGroups"));

			REQUIRE_FALSE(s.has("nonexistent"));
			REQUIRE_FALSE(s.has("group.nonexistent"));
		}
	}
}


SCENARIO(
	"`OpenMPCD::Configuration::Setting::read`"
	"")
{
	using OpenMPCD::Configuration;

	libconfig::Config config;
	config.readFile("test/data/Configuration/Setting.txt");

	GIVEN("the root setting")
	{
		const Configuration::Setting s(config.getRoot());

		THEN("`read` works for strings")
		{
			std::string str;

			s.read("str", &str);
			REQUIRE(str == "hello world");
			REQUIRE(s.read<std::string>("str") == "hello world");

			s.read("group.str", &str);
			REQUIRE(str == "foobar");
			REQUIRE(s.read<std::string>("group.str") == "foobar");

			s.read("group.group.str", &str);
			REQUIRE(str == "level3");
			REQUIRE(s.read<std::string>("group.group.str") == "level3");
		}

		THEN("`read` works for integers")
		{
			int i;

			s.read("int", &i);
			REQUIRE(i == 123);
			REQUIRE(s.read<int>("int") == 123);

			s.read("group.int", &i);
			REQUIRE(i == -789);
			REQUIRE(s.read<int>("group.int") == -789);
		}

		THEN("`read` works for `std::size_t`")
		{
			std::size_t i;

			s.read("int", &i);
			REQUIRE(i == 123);
			REQUIRE(s.read<std::size_t>("int") == 123);
		}

		THEN("`read` works for float")
		{
			float f;

			s.read("float", &f);
			REQUIRE(f == -4.56f);
			REQUIRE(s.read<float>("float") == -4.56f);

			s.read("group.float", &f);
			REQUIRE(f == -10.0f);
			REQUIRE(s.read<float>("group.float") == -10.0f);
		}

		THEN("`read` throws if given an illegal setting name")
		{
			float x;
			REQUIRE_THROWS_AS(
				s.read(".float", &x),
				OpenMPCD::InvalidArgumentException);
			REQUIRE_THROWS_AS(
				s.read("float.", &x),
				OpenMPCD::InvalidArgumentException);
			REQUIRE_THROWS_AS(
				s.read(".", &x),
				OpenMPCD::InvalidArgumentException);

			REQUIRE_THROWS_AS(
				s.read<float>(".float"),
				OpenMPCD::InvalidArgumentException);
			REQUIRE_THROWS_AS(
				s.read<float>("float."),
				OpenMPCD::InvalidArgumentException);
			REQUIRE_THROWS_AS(
				s.read<float>("."),
				OpenMPCD::InvalidArgumentException);
		}

		THEN("`read` throws if setting does not exist")
		{
			float x;
			REQUIRE_THROWS_AS(
				s.read("nonexistent", &x),
				OpenMPCD::InvalidConfigurationException);
			REQUIRE_THROWS_AS(
				s.read("group.nonexistent", &x),
				OpenMPCD::InvalidConfigurationException);

			REQUIRE_THROWS_AS(
				s.read<float>("nonexistent"),
				OpenMPCD::InvalidConfigurationException);
			REQUIRE_THROWS_AS(
				s.read<float>("group.nonexistent"),
				OpenMPCD::InvalidConfigurationException);
		}

		THEN("`read` throws if setting does not match the type")
		{
			float x;
			REQUIRE_THROWS_AS(
				s.read("int", &x),
				OpenMPCD::InvalidConfigurationException);
			REQUIRE_THROWS_AS(
				s.read("group.int", &x),
				OpenMPCD::InvalidConfigurationException);

			REQUIRE_THROWS_AS(
				s.read<float>("int"),
				OpenMPCD::InvalidConfigurationException);
			REQUIRE_THROWS_AS(
				s.read<float>("group.int"),
				OpenMPCD::InvalidConfigurationException);
		}

		#ifdef OPENMPCD_DEBUG
			THEN("`read` throws if given a `NULL` pointer")
			{
				REQUIRE_THROWS_AS(
					s.read<float>("float", NULL),
					OpenMPCD::NULLPointerException);
			}
		#endif
	}
}


SCENARIO(
	"`OpenMPCD::Configuration::Setting::getChildCount`"
	"")
{
	using OpenMPCD::Configuration;

	libconfig::Config config;
	config.readFile("test/data/Configuration/Setting.txt");

	GIVEN("the root setting")
	{
		const Configuration::Setting s(config.getRoot());

		THEN("`getChildCount` works")
		{
			REQUIRE(s.getChildCount() == 6);
			REQUIRE(s.getSetting("group").getChildCount() == 4);
			REQUIRE(s.getSetting("group.group").getChildCount() == 1);
			REQUIRE(s.getSetting("list").getChildCount() == 4);
			REQUIRE(s.getSetting("listOfGroups").getChildCount() == 2);
		}
	}
}



SCENARIO(
	"`OpenMPCD::Configuration::Setting::getChild`"
	"")
{
	using OpenMPCD::Configuration;

	libconfig::Config config;
	config.readFile("test/data/Configuration/Setting.txt");

	GIVEN("the root setting")
	{
		const Configuration::Setting s(config.getRoot());

		REQUIRE(s.getChildCount() == 6);


		THEN("`getChild` throws if arguments are invalid")
		{
			REQUIRE_THROWS_AS(
				s.getChild(s.getChildCount()),
				OpenMPCD::InvalidArgumentException);

			REQUIRE_THROWS_AS(
				s.getChild(s.getChildCount() + 1),
				OpenMPCD::InvalidArgumentException);
		}


		THEN("`getChild` works if supplied with acceptable arguments")
		{

			REQUIRE(s.getChild(0).getName() == "str");
			REQUIRE(s.getChild(1).getName() == "int");
			REQUIRE(s.getChild(2).getName() == "float");
			REQUIRE(s.getChild(3).getName() == "group");
			REQUIRE(s.getChild(4).getName() == "list");
			REQUIRE(s.getChild(5).getName() == "listOfGroups");

			REQUIRE(s.getChild(3).read<std::string>("group.str") == "level3");

			REQUIRE(s.getChild(5).getChild(1).has("isGroup2"));
		}
	}
}




SCENARIO(
	"`OpenMPCD::Configuration::Setting::getSetting`"
	"")
{
	using OpenMPCD::Configuration;

	libconfig::Config config;
	config.readFile("test/data/Configuration/Setting.txt");

	GIVEN("the root setting")
	{
		const Configuration::Setting s(config.getRoot());

		THEN("`getSetting` works")
		{
			REQUIRE(s.getSetting("group").read<std::string>("str") == "foobar");
			REQUIRE(s.getSetting("group").read<int>("int") == -789);
			REQUIRE(s.getSetting("group").read<double>("float") == -10.0);
			REQUIRE(
				s.getSetting("group.group").read<std::string>("str")
				==
				"level3");
			REQUIRE(
				s.getSetting("group").getSetting("group").read<std::string>(
					"str")
				==
				"level3");
		}


		THEN("`getSetting` throws if given an illegal setting name")
		{
			REQUIRE_THROWS_AS(
				s.getSetting(".float"),
				OpenMPCD::InvalidArgumentException);
			REQUIRE_THROWS_AS(
				s.getSetting("float."),
				OpenMPCD::InvalidArgumentException);
			REQUIRE_THROWS_AS(
				s.getSetting("."),
				OpenMPCD::InvalidArgumentException);
		}

		THEN("`getSetting` throws if setting does not exist")
		{
			REQUIRE_THROWS_AS(
				s.getSetting("nonexistent"),
				OpenMPCD::InvalidConfigurationException);
			REQUIRE_THROWS_AS(
				s.getSetting("group.nonexistent"),
				OpenMPCD::InvalidConfigurationException);
			REQUIRE_THROWS_AS(
				s.getSetting("nonexistent.nonexistent"),
				OpenMPCD::InvalidConfigurationException);
		}
	}
}


SCENARIO(
	"`OpenMPCD::Configuration::Setting::getList`"
	"")
{
	using OpenMPCD::Configuration;

	libconfig::Config config;
	config.readFile("test/data/Configuration/List.txt");

	GIVEN("the root setting")
	{
		const Configuration::Setting s(config.getRoot());

		THEN("`getList` works")
		{
			const Configuration::List emptyArray = s.getList("emptyArray");
			const Configuration::List emptyList = s.getList("emptyList");
			const Configuration::List strArray = s.getList("strArray");
			const Configuration::List strList = s.getList("strList");
			const Configuration::List intArray = s.getList("intArray");
			const Configuration::List mixedList = s.getList("mixedList");
			const Configuration::List nestedArray =
				s.getList("group.nestedArray");
			const Configuration::List nestedList =
				s.getSetting("group").getList("nestedList");
			const Configuration::List listOfGroups = s.getList("listOfGroups");

			REQUIRE(emptyArray.getSize() == 0);

			REQUIRE(emptyList.getSize() == 0);

			REQUIRE(strArray.getSize() == 2);
			REQUIRE(strArray.read<std::string>(0) == "hello");
			REQUIRE(strArray.read<std::string>(1) == "world");

			REQUIRE(strList.getSize() == 2);
			REQUIRE(strList.read<std::string>(0) == "foo");
			REQUIRE(strList.read<std::string>(1) == "bar");

			REQUIRE(intArray.getSize() == 3);
			REQUIRE(intArray.read<int>(0) == -1);
			REQUIRE(intArray.read<int>(1) == 2);
			REQUIRE(intArray.read<int>(2) == 34);

			REQUIRE(mixedList.getSize() == 2);
			REQUIRE(mixedList.read<std::string>(0) == "pi");
			REQUIRE(mixedList.read<double>(1) == 3.1415);

			REQUIRE(nestedArray.getSize() == 4);
			REQUIRE(nestedArray.read<double>(0) == 0.0);
			REQUIRE(nestedArray.read<double>(1) == 1.5);
			REQUIRE(nestedArray.read<double>(2) == 3.7);
			REQUIRE(nestedArray.read<double>(3) == -4.91);

			REQUIRE(nestedList.getSize() == 2);
			REQUIRE(nestedList.read<bool>(0) == false);
			REQUIRE(nestedList.read<std::string>(1) == "xyz");

			REQUIRE(listOfGroups.getSize() == 2);
			REQUIRE(
				listOfGroups.getSetting(0).read<std::string>("name")
				==
				"group1");
			REQUIRE(
				listOfGroups.getSetting(1).read<std::string>("name")
				==
				"group2");
			REQUIRE(listOfGroups.getSetting(1).read<bool>("isGroup2"));
		}


		THEN("`getList` throws if given an illegal setting name")
		{
			REQUIRE_THROWS_AS(
				s.getList(".float"),
				OpenMPCD::InvalidArgumentException);
			REQUIRE_THROWS_AS(
				s.getList("float."),
				OpenMPCD::InvalidArgumentException);
			REQUIRE_THROWS_AS(
				s.getList("."),
				OpenMPCD::InvalidArgumentException);
		}

		THEN("`getList` throws if setting does not exist")
		{
			REQUIRE_THROWS_AS(
				s.getList("nonexistent"),
				OpenMPCD::InvalidConfigurationException);
			REQUIRE_THROWS_AS(
				s.getList("group.nonexistent"),
				OpenMPCD::InvalidConfigurationException);
			REQUIRE_THROWS_AS(
				s.getList("nonexistent.nonexistent"),
				OpenMPCD::InvalidConfigurationException);
		}
	}
}


SCENARIO(
	"`OpenMPCD::Configuration::Setting::childrenHaveNamesInCollection`"
	"")
{
	using OpenMPCD::Configuration;

	libconfig::Config config;
	config.readFile("test/data/Configuration/Setting.txt");

	GIVEN("the root setting")
	{
		const Configuration::Setting s(config.getRoot());

		std::set<std::string> names;
		std::string offender;

		REQUIRE_FALSE(s.childrenHaveNamesInCollection(names));

		names.insert("str");
		REQUIRE_FALSE(s.childrenHaveNamesInCollection(names));

		names.insert("float");
		REQUIRE_FALSE(s.childrenHaveNamesInCollection(names, &offender));
		REQUIRE(offender == "int");

		names.insert("int");
		REQUIRE_FALSE(s.childrenHaveNamesInCollection(names));

		names.insert("group");
		REQUIRE_FALSE(s.childrenHaveNamesInCollection(names, &offender));
		REQUIRE(offender == "list");

		names.insert("list");
		REQUIRE_FALSE(s.childrenHaveNamesInCollection(names, &offender));
		REQUIRE(offender == "listOfGroups");

		names.insert("listOfGroups");
		REQUIRE(s.childrenHaveNamesInCollection(names, &offender));
		REQUIRE(offender == "listOfGroups");

		names.insert("doesNotExist");
		REQUIRE(s.childrenHaveNamesInCollection(names));

		names.insert("isGroup2");
		REQUIRE(s.childrenHaveNamesInCollection(names, &offender));
		REQUIRE(offender == "listOfGroups");

		REQUIRE(s.getSetting("group").childrenHaveNamesInCollection(names));
	}
}



SCENARIO(
	"`OpenMPCD::Configuration::List::getSetting`",
	"")
{
	using OpenMPCD::Configuration;

	libconfig::Config config;
	config.readFile("test/data/Configuration/Setting.txt");

	GIVEN("a `List` instance")
	{
		const Configuration::List list(config.lookup("listOfGroups"));

		REQUIRE(list.getSize() == 2);

		THEN("`getSetting(0)` is as expected")
		{
			const Configuration::Setting s = list.getSetting(0);

			REQUIRE_FALSE(s.hasName());
			REQUIRE(s.read<std::string>("name") == "group1");
		}

		THEN("`getSetting(1)` is as expected")
		{
			const Configuration::Setting s = list.getSetting(1);

			REQUIRE_FALSE(s.hasName());
			REQUIRE(s.read<std::string>("name") == "group2");
			REQUIRE(s.read<bool>("isGroup2"));
		}

		THEN("`getsSetting(2)` throws")
		{
			REQUIRE_THROWS_AS(
				list.getSetting(2),
				OpenMPCD::InvalidConfigurationException);
		}
	}
}


SCENARIO(
	"`OpenMPCD::Configuration::List::read`, two-parameter version"
	"")
{
	using OpenMPCD::Configuration;

	libconfig::Config config;
	config.readFile("test/data/Configuration/List.txt");

	GIVEN("`List` instances")
	{
		const Configuration::List strArray(config.lookup("strArray"));
		const Configuration::List strList(config.lookup("strList"));
		const Configuration::List intArray(config.lookup("intArray"));
		const Configuration::List mixedList(config.lookup("mixedList"));
		const Configuration::List nestedArray(
			config.lookup("group.nestedArray"));
		const Configuration::List nestedList(config.lookup("group.nestedList"));
		const Configuration::List listOfGroups(config.lookup("listOfGroups"));

		THEN("`read`, two-parameter version, works")
		{
			std::string str;
			int i;
			double d;
			bool b;

			REQUIRE(strArray.getSize() == 2);
			strArray.read(0, &str);
			REQUIRE(str == "hello");
			strArray.read(1, &str);
			REQUIRE(str == "world");

			REQUIRE(strList.getSize() == 2);
			strList.read(0, &str);
			REQUIRE(str == "foo");
			strList.read(1, &str);
			REQUIRE(str == "bar");

			REQUIRE(intArray.getSize() == 3);
			i = 0;
			intArray.read(0, &i);
			REQUIRE(i == -1);
			intArray.read(1, &i);
			REQUIRE(i == 2);
			intArray.read(2, &i);
			REQUIRE(i == 34);

			REQUIRE(mixedList.getSize() == 2);
			mixedList.read(0, &str);
			REQUIRE(str == "pi");
			d = 0;
			mixedList.read(1, &d);
			REQUIRE(d == 3.1415);

			REQUIRE(nestedArray.getSize() == 4);
			nestedArray.read(0, &d);
			REQUIRE(d == 0.0);
			nestedArray.read(1, &d);
			REQUIRE(d == 1.5);
			nestedArray.read(2, &d);
			REQUIRE(d == 3.7);
			nestedArray.read(3, &d);
			REQUIRE(d == -4.91);

			REQUIRE(nestedList.getSize() == 2);
			b = true;
			nestedList.read(0, &b);
			REQUIRE(b == false);
			nestedList.read(1, &str);
			REQUIRE(str == "xyz");

			REQUIRE(listOfGroups.getSize() == 2);
			listOfGroups.getSetting(0).read("name", &str);
			REQUIRE(str == "group1");
			listOfGroups.getSetting(1).read("name", &str);
			REQUIRE(str == "group2");
			b = false;
			listOfGroups.getSetting(1).read("isGroup2", &b);
			REQUIRE(b);
		}

		THEN("`read` throws if setting does not exist")
		{
			REQUIRE_THROWS_AS(
				strArray.read<std::string>(2),
				OpenMPCD::InvalidConfigurationException);
			REQUIRE_THROWS_AS(
				strList.read<std::string>(2),
				OpenMPCD::InvalidConfigurationException);
		}

		#ifdef OPENMPCD_DEBUG
			THEN("`read` throws if given a `nullptr`")
			{
				std::string* const null = NULL;

				REQUIRE_THROWS_AS(
					strArray.read(0, null),
					OpenMPCD::NULLPointerException);
				REQUIRE_THROWS_AS(
					strList.read(0, null),
					OpenMPCD::NULLPointerException);
			}
		#endif
	}
}


SCENARIO(
	"`OpenMPCD::Configuration::List::read`, one-parameter version"
	"")
{
	using OpenMPCD::Configuration;

	libconfig::Config config;
	config.readFile("test/data/Configuration/List.txt");

	GIVEN("`List` instances")
	{
		const Configuration::List strArray(config.lookup("strArray"));
		const Configuration::List strList(config.lookup("strList"));
		const Configuration::List intArray(config.lookup("intArray"));
		const Configuration::List mixedList(config.lookup("mixedList"));
		const Configuration::List nestedArray(
			config.lookup("group.nestedArray"));
		const Configuration::List nestedList(config.lookup("group.nestedList"));
		const Configuration::List listOfGroups(config.lookup("listOfGroups"));

		THEN("`read`, one-parameter version, works")
		{
			REQUIRE(strArray.getSize() == 2);
			REQUIRE(strArray.read<std::string>(0) == "hello");
			REQUIRE(strArray.read<std::string>(1) == "world");

			REQUIRE(strList.getSize() == 2);
			REQUIRE(strList.read<std::string>(0) == "foo");
			REQUIRE(strList.read<std::string>(1) == "bar");

			REQUIRE(intArray.getSize() == 3);
			REQUIRE(intArray.read<int>(0) == -1);
			REQUIRE(intArray.read<int>(1) == 2);
			REQUIRE(intArray.read<int>(2) == 34);

			REQUIRE(mixedList.getSize() == 2);
			REQUIRE(mixedList.read<std::string>(0) == "pi");
			REQUIRE(mixedList.read<double>(1) == 3.1415);

			REQUIRE(nestedArray.getSize() == 4);
			REQUIRE(nestedArray.read<double>(0) == 0.0);
			REQUIRE(nestedArray.read<double>(1) == 1.5);
			REQUIRE(nestedArray.read<double>(2) == 3.7);
			REQUIRE(nestedArray.read<double>(3) == -4.91);

			REQUIRE(nestedList.getSize() == 2);
			REQUIRE(nestedList.read<bool>(0) == false);
			REQUIRE(nestedList.read<std::string>(1) == "xyz");

			REQUIRE(listOfGroups.getSize() == 2);
			REQUIRE(
				listOfGroups.getSetting(0).read<std::string>("name")
				==
				"group1");
			REQUIRE(
				listOfGroups.getSetting(1).read<std::string>("name")
				==
				"group2");
			REQUIRE(listOfGroups.getSetting(1).read<bool>("isGroup2"));
		}

		THEN("`read` throws if setting does not exist")
		{
			REQUIRE_THROWS_AS(
				strArray.read<std::string>(2),
				OpenMPCD::InvalidConfigurationException);
			REQUIRE_THROWS_AS(
				strList.read<std::string>(2),
				OpenMPCD::InvalidConfigurationException);
		}
	}
}


SCENARIO(
	"`OpenMPCD::Configuration` basics",
	"")
{
	GIVEN("an empty configuration instance")
	{
		OpenMPCD::Configuration config;

		WHEN("a new top-level, non-group setting is set")
		{
			config.set("toplevel", int(1));

			THEN("`has` returns `true`")
			{
				REQUIRE(config.has("toplevel"));
			}

			AND_THEN("it can be retrieved via `read` via a pointer")
			{
				int val;
				config.read("toplevel", &val);
				REQUIRE(val == 1);
			}

			AND_THEN("it can be retrieved via `read` via the return value")
			{
				REQUIRE(config.read<int>("toplevel") == 1);
			}

			AND_THEN("`assertValue` does not throw")
			{
				REQUIRE_NOTHROW(config.assertValue("toplevel", int(1)));
			}
		}

		WHEN("a new, non-top-level setting is set, whose parent does not exist yet")
		{
			config.set("top.sublevel", "qwer");

			THEN("`has` returns `true`")
			{
				REQUIRE(config.has("top.sublevel"));
			}

			AND_THEN("it can be retrieved via `read` via a pointer")
			{
				std::string val;
				config.read("top.sublevel", &val);
				REQUIRE(val == "qwer");
			}

			AND_THEN("it can be retrieved via `read` via the return value")
			{
				REQUIRE(config.read<std::string>("top.sublevel") == "qwer");
			}

			AND_THEN("`assertValue` does not throw")
			{
				REQUIRE_NOTHROW(config.assertValue("top.sublevel", "qwer"));
			}
		}
	}
}


SCENARIO(
	"`OpenMPCD::Configuration::Configuration(const std::string&)`",
	"")
{
	static const char* const validFilePath =
		"test/data/Configuration/Setting.txt";
	static const char* const nonExistantFilePath =
		"test/data/Configuration/nonExistantFile.txt";
	static const char* const malformedFilePath =
		"test/data/Configuration/malformed.txt";

	GIVEN("a valid configuration file that exists")
	{
		REQUIRE(boost::filesystem::is_regular_file(validFilePath));

		THEN("it can be constructed from")
		{
			OpenMPCD::Configuration config(validFilePath);

			AND_THEN("its values are read correctly")
			{
				REQUIRE(config.read<int>("int") == 123);
				REQUIRE(config.read<std::string>("group.str") == "foobar");
			}
		}
	}

	GIVEN("a non-existent configuration file")
	{
		REQUIRE_FALSE(boost::filesystem::is_regular_file(nonExistantFilePath));

		THEN("the constructor throws")
		{
			REQUIRE_THROWS_AS(
				OpenMPCD::Configuration config(nonExistantFilePath),
				OpenMPCD::IOException);
		}
	}

	GIVEN("a malformed configuration file")
	{
		REQUIRE(boost::filesystem::is_regular_file(malformedFilePath));

		THEN("the constructor throws")
		{
			REQUIRE_THROWS_AS(
				OpenMPCD::Configuration config(malformedFilePath),
				OpenMPCD::MalformedFileException);
		}
	}
}


SCENARIO(
	"`OpenMPCD::Configuration` copy constructor",
	"")
{
	GIVEN("an empty configuration instance")
	{
		OpenMPCD::Configuration config;

		THEN("a copy can be made")
		{
			OpenMPCD::Configuration copy(config);
		}
	}

	GIVEN("a non-empty configuration instance")
	{
		OpenMPCD::Configuration config;

		config.set("toplevel", int(1));
		config.set("another.sub", "asdf");

		WHEN("a copy is created")
		{
			OpenMPCD::Configuration copy(config);

			THEN("it has the same settings")
			{
				REQUIRE(copy.read<int>("toplevel") == 1);
				REQUIRE(copy.read<std::string>("another.sub") == "asdf");
			}
		}
	}
}


SCENARIO(
	"`OpenMPCD::Configuration::getSetting`",
	"")
{
	static const char* const filePath = "test/data/Configuration/Setting.txt";

	using OpenMPCD::Configuration;
	typedef Configuration::Setting Setting;

	GIVEN("a `Configuration` instance")
	{
		const Configuration config(filePath);

		THEN("`getSetting` works for simple settings")
		{
			const Setting str = config.getSetting("group.group.str");
			const Setting i = config.getSetting("group.int");
			const Setting f = config.getSetting("float");

			REQUIRE(str.hasName());
			REQUIRE(str.getName() == "str");

			REQUIRE(i.hasName());
			REQUIRE(i.getName() == "int");

			REQUIRE(f.hasName());
			REQUIRE(f.getName() == "float");
		}

		THEN("`getSetting` works for aggregate settings")
		{
			const Setting group = config.getSetting("group");

			REQUIRE(group.read<std::string>("str") == "foobar");
		}

		THEN("`getSetting` throws for unknown settings")
		{
			REQUIRE_THROWS_AS(
				config.getSetting("does_not_exist"),
				OpenMPCD::InvalidConfigurationException);
		}
	}
}


SCENARIO(
	"`OpenMPCD::Configuration::set`",
	"")
{
	GIVEN("a `Configuration` instance that is empty")
	{
		OpenMPCD::Configuration config;

		THEN("one can create new top-level settings")
		{
			config.set("toplevel", 0);
		}

		THEN("one can create new sub-level settings")
		{
			config.set("toplevel.sub", "test");
		}

		THEN("one can create variables of type `unsigned int`")
		{
			config.set("unsigned-int", static_cast<unsigned int>(1));
		}
	}
}

SCENARIO(
	"`OpenMPCD::Configuration::createGroup`",
	"")
{
	GIVEN("an empty `Configuration` instance")
	{
		OpenMPCD::Configuration config;

		THEN("one can create new top-level groups")
		{
			config.createGroup("toplevel");

			AND_THEN("one can create direct child groups")
			{
				config.createGroup("toplevel.sub1");
				config.createGroup("toplevel.sub2");
			}

			AND_THEN("one can create indirect child groups")
			{
				config.createGroup("toplevel.indirect.child1");
				config.createGroup("toplevel.indirect.child2");
			}
		}

		THEN("one can create new sub-level settings directly")
		{
			config.createGroup("top-indirect.sub");
			config.createGroup("top-indirect-manylayers.a.b.c.d.e.f");

			AND_THEN("one can create ordinary settings beneath")
			{
				config.set("top-indirect-manylayers.a.b.c.d.e.f.stringvar", "test");
			}
		}
	}
}


SCENARIO(
	"`OpenMPCD::Configuration::createList`",
	"")
{
	GIVEN("an empty `Configuration` instance")
	{
		OpenMPCD::Configuration config;

		THEN("one can create new top-level lists")
		{
			config.createList("toplevel");
		}

		THEN("one can create new sub-level lists directly")
		{
			config.createGroup("top-indirect.sub");
			config.createGroup("top-indirect-manylayers.a.b.c.d.e.f");
		}
	}
}


SCENARIO(
	"`OpenMPCD::Configuration::operator=`",
	"")
{
	GIVEN("an empty configuration instance")
	{
		OpenMPCD::Configuration config;

		WHEN("an empty instance is assigned the empty one")
		{
			OpenMPCD::Configuration alsoEmpty;

			THEN("nothing happens, and the instances are not coupled")
			{
				alsoEmpty = config;

				REQUIRE_FALSE(alsoEmpty.has("test"));
				REQUIRE_FALSE(config.has("test"));

				alsoEmpty.createGroup("test");
				REQUIRE(alsoEmpty.has("test"));
				REQUIRE_FALSE(config.has("test"));
			}
		}

		WHEN("a non-empty instance is assigned the empty one")
		{
			OpenMPCD::Configuration initialized;

			initialized.createGroup("test");

			REQUIRE(initialized.has("test"));

			initialized = config;

			THEN("it is also empty, but not coupled")
			{
				REQUIRE_FALSE(initialized.has("test"));
				REQUIRE_FALSE(config.has("test"));

				initialized.createGroup("test");
				REQUIRE(initialized.has("test"));
				REQUIRE_FALSE(config.has("test"));
			}
		}
	}

	GIVEN("a non-empty configuration instance")
	{
		OpenMPCD::Configuration config;

		config.set("toplevel", int(1));
		config.set("another.sub", "asdf");

		WHEN("it is assigned to another, empty instance")
		{
			OpenMPCD::Configuration another;

			another = config;

			THEN("they have the same settings, but are not coupled")
			{
				REQUIRE(another.read<int>("toplevel") == 1);
				REQUIRE(another.read<std::string>("another.sub") == "asdf");

				another.set("toplevel", int(2));
				REQUIRE(config.read<int>("toplevel") == 1);
				REQUIRE(another.read<int>("toplevel") == 2);
			}

			THEN("the correct reference is returned")
			{
				REQUIRE(&(another = config) == &another);
			}

			THEN("it can be assigned multiple times")
			{
				another = config;

				REQUIRE(another.read<int>("toplevel") == 1);
				REQUIRE(another.read<std::string>("another.sub") == "asdf");
			}
		}

		WHEN("it is assigned to a non-empty instance")
		{
			AND_WHEN("a setting of the correct type already exists")
			{
				OpenMPCD::Configuration another;
				another.set("toplevel", int(2));

				THEN("the values are the same after assignment")
				{
					another = config;
					REQUIRE(another.read<int>("toplevel") == 1);
				}
			}

			AND_WHEN("a setting of a different type already exists")
			{
				OpenMPCD::Configuration another;
				another.set("toplevel", "hello");

				THEN("the values are the same after assignment")
				{
					another = config;
					REQUIRE(another.read<int>("toplevel") == 1);
				}
			}

			AND_WHEN("a setting of another name already exists")
			{
				OpenMPCD::Configuration another;
				another.set("foobar", "barfoo");

				THEN("it is gone after assignment")
				{
					another = config;
					REQUIRE_FALSE(another.has("foobar"));
				}
			}
		}

		THEN("it can be set to itself and keep its data")
		{
			config = config;

			REQUIRE(config.read<int>("toplevel") == 1);
			REQUIRE(config.read<std::string>("another.sub") == "asdf");
		}
	}
}
