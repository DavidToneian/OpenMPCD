/**
 * @file
 * Tests functionality of `OpenMPCD::FlowProfile`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/FlowProfile.hpp>

#include <boost/filesystem.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <fstream>


#define SEQ_DATATYPE (float)(double)(OpenMPCD::FP)

template<typename T>
static void test_constructor()
{
	OpenMPCD::Configuration validConfig;
	validConfig.set("instrumentation.flowProfile.cellSubdivision.x", 1);
	validConfig.set("instrumentation.flowProfile.cellSubdivision.y", 1);
	validConfig.set("instrumentation.flowProfile.cellSubdivision.z", 1);
	OpenMPCD::Configuration::Setting validSettings =
		validConfig.getSetting("instrumentation.flowProfile");


	OpenMPCD::Configuration emptyConfig;
	emptyConfig.createGroup("instrumentation.flowProfile");
	OpenMPCD::Configuration::Setting emptySettings =
		validConfig.getSetting("instrumentation.flowProfile");


	OpenMPCD::Configuration invalidXConfig;
	invalidXConfig.set("instrumentation.flowProfile.cellSubdivision.x", 0);
	invalidXConfig.set("instrumentation.flowProfile.cellSubdivision.y", 1);
	invalidXConfig.set("instrumentation.flowProfile.cellSubdivision.z", 1);
	OpenMPCD::Configuration::Setting invalidXSettings =
		invalidXConfig.getSetting("instrumentation.flowProfile");

	OpenMPCD::Configuration invalidYConfig;
	invalidYConfig.set("instrumentation.flowProfile.cellSubdivision.x", 1);
	invalidYConfig.set("instrumentation.flowProfile.cellSubdivision.y", 0);
	invalidYConfig.set("instrumentation.flowProfile.cellSubdivision.z", 1);
	OpenMPCD::Configuration::Setting invalidYSettings =
		invalidYConfig.getSetting("instrumentation.flowProfile");


	OpenMPCD::Configuration invalidZConfig;
	invalidZConfig.set("instrumentation.flowProfile.cellSubdivision.x", 1);
	invalidZConfig.set("instrumentation.flowProfile.cellSubdivision.y", 1);
	invalidZConfig.set("instrumentation.flowProfile.cellSubdivision.z", 0);
	OpenMPCD::Configuration::Setting invalidZSettings =
		invalidZConfig.getSetting("instrumentation.flowProfile");

	OpenMPCD::Configuration partialConfig;
	partialConfig.set("instrumentation.flowProfile.cellSubdivision.x", 1);
	partialConfig.set("instrumentation.flowProfile.cellSubdivision.z", 0);
	OpenMPCD::Configuration::Setting partialSettings =
		partialConfig.getSetting("instrumentation.flowProfile");




	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			OpenMPCD::FlowProfile<T>(0, 1, 1, validSettings),
			OpenMPCD::InvalidArgumentException);
		REQUIRE_THROWS_AS(
			OpenMPCD::FlowProfile<T>(1, 0, 1, validSettings),
			OpenMPCD::InvalidArgumentException);
		REQUIRE_THROWS_AS(
			OpenMPCD::FlowProfile<T>(1, 1, 0, validSettings),
			OpenMPCD::InvalidArgumentException);
	#endif

	REQUIRE_THROWS_AS(
		OpenMPCD::FlowProfile<T>(1, 1, 1, invalidXSettings),
		OpenMPCD::InvalidConfigurationException);
	REQUIRE_THROWS_AS(
		OpenMPCD::FlowProfile<T>(1, 1, 1, invalidYSettings),
		OpenMPCD::InvalidConfigurationException);
	REQUIRE_THROWS_AS(
		OpenMPCD::FlowProfile<T>(1, 1, 1, invalidZSettings),
		OpenMPCD::InvalidConfigurationException);
	REQUIRE_THROWS_AS(
		OpenMPCD::FlowProfile<T>(1, 1, 1, partialSettings),
		OpenMPCD::InvalidConfigurationException);


	REQUIRE_NOTHROW(OpenMPCD::FlowProfile<T>(1, 1, 1, validSettings));
	REQUIRE_NOTHROW(OpenMPCD::FlowProfile<T>(1, 1, 1, emptySettings));
}
SCENARIO(
	"`OpenMPCD::FlowProfile::FlowProfile`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_constructor<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}


template<typename T>
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
		config.set("instrumentation.flowProfile.cellSubdivision.x", cellSubX);
		config.set("instrumentation.flowProfile.cellSubdivision.y", cellSubY);
		config.set("instrumentation.flowProfile.cellSubdivision.z", cellSubZ);
		OpenMPCD::Configuration::Setting settings =
			config.getSetting("instrumentation.flowProfile");


		OpenMPCD::FlowProfile<T> fp(
			boxSizeX, boxSizeY, boxSizeX,
			settings);

		REQUIRE(fp.getCellSubdivisionsX() == cellSubX);
		REQUIRE(fp.getCellSubdivisionsY() == cellSubY);
		REQUIRE(fp.getCellSubdivisionsZ() == cellSubZ);
	}
}
SCENARIO(
	"`OpenMPCD::FlowProfile::getCellSubdivisionsX`, "
	"`OpenMPCD::FlowProfile::getCellSubdivisionsY`, "
	"`OpenMPCD::FlowProfile::getCellSubdivisionsZ`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_getCellSubdivisionsX_getCellSubdivisionsY_getCellSubdivisionsZ \
			<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}


template<typename T>
static void test_add()
{
	OpenMPCD::Configuration config;
	config.set("instrumentation.flowProfile.cellSubdivision.x", 4);
	config.set("instrumentation.flowProfile.cellSubdivision.y", 5);
	config.set("instrumentation.flowProfile.cellSubdivision.z", 6);
	OpenMPCD::Configuration::Setting settings =
		config.getSetting("instrumentation.flowProfile");

	OpenMPCD::FlowProfile<T> fp(1, 2, 3, settings);
	const OpenMPCD::Vector3D<T> v(0, 0, 0);

	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			fp.add(0, 0, 0, v),
			OpenMPCD::InvalidCallException);
	#endif

	fp.newSweep();

	REQUIRE_NOTHROW(fp.add(0, 0, 0, v));
	REQUIRE_NOTHROW(fp.add(3, 9, 17, v));

	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			fp.add(4, 0, 0, v),
			OpenMPCD::OutOfBoundsException);
		REQUIRE_THROWS_AS(
			fp.add(0, 10, 0, v),
			OpenMPCD::OutOfBoundsException);
		REQUIRE_THROWS_AS(
			fp.add(0, 0, 18, v),
			OpenMPCD::OutOfBoundsException);
	#endif
}
SCENARIO(
	"`OpenMPCD::FlowProfile::add`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_add<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}


template<typename T>
static void test_saveToFile()
{
	static const unsigned int boxX = 2;
	static const unsigned int boxY = 3;
	static const unsigned int boxZ = 4;
	static const unsigned int cellX = 5;
	static const unsigned int cellY = 6;
	static const unsigned int cellZ = 7;
	static const std::size_t valueCount = 5;


	OpenMPCD::Configuration config;
	config.set("instrumentation.flowProfile.cellSubdivision.x", cellX);
	config.set("instrumentation.flowProfile.cellSubdivision.y", cellY);
	config.set("instrumentation.flowProfile.cellSubdivision.z", cellZ);
	OpenMPCD::Configuration::Setting settings =
		config.getSetting("instrumentation.flowProfile");


	OpenMPCD::FlowProfile<T> fp(boxX, boxY, boxZ, settings);
	fp.newSweep();

	boost::multi_array<OpenMPCD::OnTheFlyStatistics<T>, 4> points(
		boost::extents[boxX * cellX][boxY * cellY][boxZ * cellZ][3]);


	for(std::size_t x = 0; x < boxX * cellX; ++x)
	{
		for(std::size_t y = 0; y < boxY * cellY; ++y)
		{
			for(std::size_t z = 0; z < boxZ * cellZ; ++z)
			{
				for(std::size_t i = 0; i < valueCount; ++i)
				{
					const OpenMPCD::Vector3D<T> v(i * x, i * y, i * z);

					fp.add(x, y, z, v);

					points[x][y][z][0].addDatum(v.getX());
					points[x][y][z][1].addDatum(v.getY());
					points[x][y][z][2].addDatum(v.getZ());
				}
			}
		}
	}

	const boost::filesystem::path tempdir =
		boost::filesystem::temp_directory_path() /
		boost::filesystem::unique_path();
	boost::filesystem::create_directories(tempdir);

	fp.saveToFile((tempdir / "flowProfile.data").c_str());



	std::ifstream file((tempdir / "flowProfile.data").c_str());

	std::string expectedHeader;
	expectedHeader += "#";
	expectedHeader += "x\t";
	expectedHeader += "y\t";
	expectedHeader += "z\t";
	expectedHeader += "meanVelX\t";
	expectedHeader += "meanVelY\t";
	expectedHeader += "meanVelZ\t";
	expectedHeader += "stdDevVelX\t";
	expectedHeader += "stdDevVelY\t";
	expectedHeader += "stdDevVelZ\t";
	expectedHeader += "sampleSize";
	std::string header;
	std::getline(file, header);
	REQUIRE(header == expectedHeader);

	T posX, posY, posZ;
	T velX, velY, velZ;
	T stdVelX, stdVelY, stdVelZ;
	unsigned long int sampleSize;

	while(
		file
		>> posX >> posY >> posZ
		>> velX >> velY >> velZ
		>> stdVelX >> stdVelY >> stdVelZ
		>> sampleSize)
	{
		const long int x = boost::math::lround(posX * cellX);
		const long int y = boost::math::lround(posY * cellY);
		const long int z = boost::math::lround(posZ * cellZ);

		REQUIRE(velX == points[x][y][z][0].getSampleMean());
		REQUIRE(velY == points[x][y][z][1].getSampleMean());
		REQUIRE(velZ == points[x][y][z][2].getSampleMean());

		REQUIRE(stdVelX == points[x][y][z][0].getSampleStandardDeviation());
		REQUIRE(stdVelY == points[x][y][z][1].getSampleStandardDeviation());
		REQUIRE(stdVelZ == points[x][y][z][2].getSampleStandardDeviation());

		REQUIRE(sampleSize == points[x][y][z][0].getSampleSize());
		REQUIRE(sampleSize == points[x][y][z][1].getSampleSize());
		REQUIRE(sampleSize == points[x][y][z][2].getSampleSize());
	}

	std::string tmp;
	file >> tmp;
	REQUIRE(file.eof());

	boost::filesystem::remove_all(tempdir);
}
SCENARIO(
	"`OpenMPCD::FlowProfile::saveToFile`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_saveToFile<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}
