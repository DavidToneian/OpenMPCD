/**
 * @file
 * Tests functionality of `OpenMPCD::DensityProfile`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/DensityProfile.hpp>

#include <boost/filesystem.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/multi_array.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <fstream>


#define SEQ_DATATYPE (void)
//#define SEQ_DATATYPE (float)(double)(OpenMPCD::FP)

//template<typename T>
static void test_constructor()
{
	typedef OpenMPCD::DensityProfile DP;

	OpenMPCD::Configuration validConfig;
	validConfig.set("instrumentation.densityProfile.cellSubdivision.x", 1);
	validConfig.set("instrumentation.densityProfile.cellSubdivision.y", 1);
	validConfig.set("instrumentation.densityProfile.cellSubdivision.z", 1);
	OpenMPCD::Configuration::Setting validSettings =
		validConfig.getSetting("instrumentation.densityProfile");


	OpenMPCD::Configuration emptyConfig;
	emptyConfig.createGroup("instrumentation.densityProfile");
	OpenMPCD::Configuration::Setting emptySettings =
		validConfig.getSetting("instrumentation.densityProfile");


	OpenMPCD::Configuration invalidXConfig;
	invalidXConfig.set("instrumentation.densityProfile.cellSubdivision.x", 0);
	invalidXConfig.set("instrumentation.densityProfile.cellSubdivision.y", 1);
	invalidXConfig.set("instrumentation.densityProfile.cellSubdivision.z", 1);
	OpenMPCD::Configuration::Setting invalidXSettings =
		invalidXConfig.getSetting("instrumentation.densityProfile");

	OpenMPCD::Configuration invalidYConfig;
	invalidYConfig.set("instrumentation.densityProfile.cellSubdivision.x", 1);
	invalidYConfig.set("instrumentation.densityProfile.cellSubdivision.y", 0);
	invalidYConfig.set("instrumentation.densityProfile.cellSubdivision.z", 1);
	OpenMPCD::Configuration::Setting invalidYSettings =
		invalidYConfig.getSetting("instrumentation.densityProfile");


	OpenMPCD::Configuration invalidZConfig;
	invalidZConfig.set("instrumentation.densityProfile.cellSubdivision.x", 1);
	invalidZConfig.set("instrumentation.densityProfile.cellSubdivision.y", 1);
	invalidZConfig.set("instrumentation.densityProfile.cellSubdivision.z", 0);
	OpenMPCD::Configuration::Setting invalidZSettings =
		invalidZConfig.getSetting("instrumentation.densityProfile");

	OpenMPCD::Configuration partialConfig;
	partialConfig.set("instrumentation.densityProfile.cellSubdivision.x", 1);
	partialConfig.set("instrumentation.densityProfile.cellSubdivision.z", 0);
	OpenMPCD::Configuration::Setting partialSettings =
		partialConfig.getSetting("instrumentation.densityProfile");



	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			DP(0, 1, 1, validSettings),
			OpenMPCD::InvalidArgumentException);
		REQUIRE_THROWS_AS(
			DP(1, 0, 1, validSettings),
			OpenMPCD::InvalidArgumentException);
		REQUIRE_THROWS_AS(
			DP(1, 1, 0, validSettings),
			OpenMPCD::InvalidArgumentException);
	#endif

	REQUIRE_THROWS_AS(
		DP(1, 1, 1, invalidXSettings),
		OpenMPCD::InvalidConfigurationException);
	REQUIRE_THROWS_AS(
		DP(1, 1, 1, invalidYSettings),
		OpenMPCD::InvalidConfigurationException);
	REQUIRE_THROWS_AS(
		DP(1, 1, 1, invalidZSettings),
		OpenMPCD::InvalidConfigurationException);
	REQUIRE_THROWS_AS(
		DP(1, 1, 1, partialSettings),
		OpenMPCD::InvalidConfigurationException);


	REQUIRE_NOTHROW(DP(1, 1, 1, validSettings));
	REQUIRE_NOTHROW(DP(1, 1, 1, emptySettings));
}
SCENARIO(
	"`OpenMPCD::DensityProfile::DensityProfile`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_constructor/*<T>*/();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}



//template<typename T>
static void
test_getCellSubdivisionsX_getCellSubdivisionsY_getCellSubdivisionsZ()
{
	static const std::size_t testCount = 10;

	boost::random::mt19937 rng;
	boost::random::uniform_int_distribution<unsigned int> dist(1, 10);

	for(std::size_t i = 0; i < testCount; ++i)
	{
		const unsigned int boxSizeX = dist(rng);
		const unsigned int boxSizeY = dist(rng);
		const unsigned int boxSizeZ = dist(rng);
		const unsigned int cellSubX = dist(rng);
		const unsigned int cellSubY = dist(rng);
		const unsigned int cellSubZ = dist(rng);


		OpenMPCD::Configuration config;
		config.set(
			"instrumentation.densityProfile.cellSubdivision.x", cellSubX);
		config.set(
			"instrumentation.densityProfile.cellSubdivision.y", cellSubY);
		config.set(
			"instrumentation.densityProfile.cellSubdivision.z", cellSubZ);
		OpenMPCD::Configuration::Setting settings =
			config.getSetting("instrumentation.densityProfile");


		OpenMPCD::DensityProfile dp(
			boxSizeX, boxSizeY, boxSizeX,
			settings);

		REQUIRE(dp.getCellSubdivisionsX() == cellSubX);
		REQUIRE(dp.getCellSubdivisionsY() == cellSubY);
		REQUIRE(dp.getCellSubdivisionsZ() == cellSubZ);
	}
}
SCENARIO(
	"`OpenMPCD::DensityProfile::getCellSubdivisionsX`, "
	"`OpenMPCD::DensityProfile::getCellSubdivisionsY`, "
	"`OpenMPCD::DensityProfile::getCellSubdivisionsZ`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_getCellSubdivisionsX_getCellSubdivisionsY_getCellSubdivisionsZ \
			/*<T>*/();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}


//template<typename T>
static void test_add()
{
	OpenMPCD::Configuration config;
	config.set("instrumentation.densityProfile.cellSubdivision.x", 4);
	config.set("instrumentation.densityProfile.cellSubdivision.y", 5);
	config.set("instrumentation.densityProfile.cellSubdivision.z", 6);
	OpenMPCD::Configuration::Setting settings =
		config.getSetting("instrumentation.densityProfile");

	OpenMPCD::DensityProfile dp(1, 2, 3, settings);

	REQUIRE_NOTHROW(dp.add(0, 0, 0, 0.5));
	REQUIRE_NOTHROW(dp.add(3, 9, 17, 1.0));

	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			dp.add(4, 0, 0, 1),
			OpenMPCD::OutOfBoundsException);
		REQUIRE_THROWS_AS(
			dp.add(0, 10, 0, 1),
			OpenMPCD::OutOfBoundsException);
		REQUIRE_THROWS_AS(
			dp.add(0, 0, 18, 1),
			OpenMPCD::OutOfBoundsException);
	#endif
}
SCENARIO(
	"`OpenMPCD::DensityProfile::add`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_add/*<T>*/();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}


//template<typename T>
static void test_saveToFile()
{
	typedef OpenMPCD::FP T;

	static const unsigned int boxX = 2;
	static const unsigned int boxY = 3;
	static const unsigned int boxZ = 4;
	static const unsigned int cellX = 5;
	static const unsigned int cellY = 6;
	static const unsigned int cellZ = 7;
	static const std::size_t valueCount = 5;


	OpenMPCD::Configuration config;
	config.set("instrumentation.densityProfile.cellSubdivision.x", cellX);
	config.set("instrumentation.densityProfile.cellSubdivision.y", cellY);
	config.set("instrumentation.densityProfile.cellSubdivision.z", cellZ);
	OpenMPCD::Configuration::Setting settings =
		config.getSetting("instrumentation.densityProfile");


	OpenMPCD::DensityProfile dp(boxX, boxY, boxZ, settings);

	boost::multi_array<T, 3> points(
		boost::extents[boxX * cellX][boxY * cellY][boxZ * cellZ]);


	for(std::size_t x = 0; x < boxX * cellX; ++x)
	{
		for(std::size_t y = 0; y < boxY * cellY; ++y)
		{
			for(std::size_t z = 0; z < boxZ * cellZ; ++z)
			{
				for(std::size_t i = 0; i < valueCount; ++i)
				{
					const T mass = x + 0.1 * y + 0.01 * z + 0.001 * i;

					dp.add(x, y, z, mass);

					points[x][y][z] += mass;
				}
			}
		}
	}

	for(std::size_t i = 0; i < valueCount; ++i)
		dp.incrementFillCount();

	const boost::filesystem::path tempdir =
		boost::filesystem::temp_directory_path() /
		boost::filesystem::unique_path();
	boost::filesystem::create_directories(tempdir);

	dp.saveToFile((tempdir / "densityProfile.data").c_str());



	std::ifstream file((tempdir / "densityProfile.data").c_str());


	T posX, posY, posZ;
	T density;

	while(
		file
		>> posX >> posY >> posZ
		>> density)
	{
		const long int x = boost::math::lround(posX * cellX);
		const long int y = boost::math::lround(posY * cellY);
		const long int z = boost::math::lround(posZ * cellZ);

		REQUIRE(density == points[x][y][z] / valueCount);
	}

	std::string tmp;
	file >> tmp;
	REQUIRE(file.eof());

	boost::filesystem::remove_all(tempdir);
}
SCENARIO(
	"`OpenMPCD::DensityProfile::saveToFile`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_saveToFile/*<T>*/();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}
