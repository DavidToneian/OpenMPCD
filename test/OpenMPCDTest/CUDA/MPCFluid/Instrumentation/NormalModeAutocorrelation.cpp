/**
 * @file
 * Tests
 * `OpenMPCD::CUDA::MPCFluid::Instrumentation::NormalModeAutocorrelation`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/NormalModeAutocorrelation.hpp>

#include <OpenMPCDTest/CUDA/SimulationTest.hpp>

#include <OpenMPCD/CUDA/DeviceBuffer.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Simple.hpp>
#include <OpenMPCD/CUDA/NormalMode.hpp>

#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>

#include <limits>


static const std::string configGroupName =
	"instrumentation.normalModeAutocorrelation";

class NonConstantTestFluid : public OpenMPCD::CUDA::MPCFluid::Simple
{
public:
	NonConstantTestFluid(
		const OpenMPCD::CUDA::Simulation* const sim, const unsigned int count,
		const OpenMPCD::FP streamingTimestep_, OpenMPCD::RNG& rng_,
		OpenMPCD::CUDA::DeviceMemoryManager* const devMemMgr)
		: Simple(sim, count, streamingTimestep_, rng_, devMemMgr)
	{
	}

	virtual bool numberOfParticlesPerLogicalEntityIsConstant() const
	{
		return false;
	}
};


static
const std::vector<OpenMPCD::Configuration>
getInvalidConfigurations()
{
	std::vector<OpenMPCD::Configuration> ret;
	const std::string prefix = configGroupName + ".";

	{
		OpenMPCD::Configuration config;

		config.set(prefix + "measureEveryNthSweep", 1.0);
		ret.push_back(config);
	}
	{
		OpenMPCD::Configuration config;

		config.set(prefix + "measureEveryNthSweep", 0);
		ret.push_back(config);

		config.set(prefix + "measureEveryNthSweep", -1);
		ret.push_back(config);
	}
	{
		OpenMPCD::Configuration config;
		config.set(prefix + "autocorrelationArgumentCount", 3);

		config.set(prefix + "measureEveryNthSweep", 1.0);
		ret.push_back(config);
	}
	{
		OpenMPCD::Configuration config;
		config.set(prefix + "autocorrelationArgumentCount", 3);

		config.set(prefix + "measureEveryNthSweep", 0);
		ret.push_back(config);

		config.set(prefix + "measureEveryNthSweep", -1);
		ret.push_back(config);
	}

	{
		OpenMPCD::Configuration config;

		config.set(prefix + "autocorrelationArgumentCount", 1.0);
		ret.push_back(config);
	}
	{
		OpenMPCD::Configuration config;

		config.set(prefix + "autocorrelationArgumentCount", 0);
		ret.push_back(config);

		config.set(prefix + "autocorrelationArgumentCount", -1);
		ret.push_back(config);
	}
	{
		OpenMPCD::Configuration config;
		config.set(prefix + "measureEveryNthSweep", 5);

		config.set(prefix + "autocorrelationArgumentCount", 1.0);
		ret.push_back(config);
	}
	{
		OpenMPCD::Configuration config;
		config.set(prefix + "measureEveryNthSweep", 5);

		config.set(prefix + "autocorrelationArgumentCount", 0);
		ret.push_back(config);

		config.set(prefix + "autocorrelationArgumentCount", -1);
		ret.push_back(config);
	}

	return ret;
}


static void getTestcase(
	const unsigned int measureEveryNthSweep,
	const unsigned int autocorrelationArgumentCount,
	const unsigned int chainLength,
	const unsigned int measurementCount,
	double shift,
	boost::shared_ptr<OpenMPCD::CUDA::Simulation>& simulation,
	boost::shared_ptr<
		OpenMPCD::CUDA::MPCFluid::Instrumentation::NormalModeAutocorrelation>&
		nma,
	std::vector<
		std::pair<
			std::pair<unsigned int, unsigned int>,
			std::vector<OpenMPCD::MPCParticlePositionType>
			>
		>& result)
{
	const unsigned int normalModeCount = chainLength + 1;

	OpenMPCD::Configuration config;
	config.set("initialization.particleDensity", 10);
	config.set("initialization.particleVelocityDistribution.mean", 0.0);
	config.set(
		"initialization.particleVelocityDistribution.standardDeviation", 1.0);

	config.set("mpc.simulationBoxSize.x", 10);
	config.set("mpc.simulationBoxSize.y", 10);
	config.set("mpc.simulationBoxSize.z", 10);
	config.set("mpc.timestep",            0.1);
	config.set("mpc.srdCollisionAngle",   2.27);
	config.set("mpc.gridShiftScale",      1.0);

	config.set("bulkThermostat.targetkT", 1.0);

	config.set("mpc.sweepSize", 2);

	config.set("bulkThermostat.type", "MBS");

	config.set("boundaryConditions.LeesEdwards.shearRate", 0.0);

	config.set("mpc.fluid.gaussianChains.springConstant", 1.0);
	config.set("mpc.fluid.gaussianChains.particlesPerChain", chainLength);
	config.set("mpc.fluid.gaussianChains.mdStepCount", 2);

	config.set(
		configGroupName + ".measureEveryNthSweep",
		measureEveryNthSweep);
	config.set(
		configGroupName + ".autocorrelationArgumentCount",
		autocorrelationArgumentCount);

	if(std::isnan(shift))
	{
		shift = 0;
	}
	else
	{
		config.set(configGroupName + ".shift", shift);
	}


	simulation.reset(new OpenMPCD::CUDA::Simulation(config, 0));
	const OpenMPCD::CUDA::MPCFluid::Base* const fluid =
		&simulation->getMPCFluid();

	const std::size_t bufferElementCount =
		fluid->getNumberOfLogicalEntities() * normalModeCount;


	nma.reset(
		new OpenMPCD::CUDA::MPCFluid::Instrumentation::NormalModeAutocorrelation(
			config, fluid));


	using OpenMPCD::CUDA::DeviceBuffer;
	typedef OpenMPCD::MPCParticlePositionType T;
	std::deque<boost::shared_ptr<DeviceBuffer<T> > > snapshots;

	for(unsigned int m = 0; m < measureEveryNthSweep * measurementCount; ++m)
	{
		nma->measure();

		boost::shared_ptr<DeviceBuffer<T> > snapshot(
			new DeviceBuffer<T>(bufferElementCount));
		snapshots.push_back(snapshot);

		OpenMPCD::CUDA::NormalMode::computeNormalCoordinates(
			chainLength,
			fluid->getNumberOfLogicalEntities(),
			fluid->getDevicePositions(),
			snapshot->getPointer(),
			shift);

		simulation->sweep();
	}

	typedef std::pair<unsigned int, unsigned int> Times;
	typedef std::vector<T> Values;

	using OpenMPCD::CUDA::NormalMode::getAverageNormalCoordinateAutocorrelation;

	for(
		unsigned int stride = 0;
		stride < autocorrelationArgumentCount;
		++stride)
	{
		for(
			unsigned int time0 = 0;
			time0 < snapshots.size();
			time0 += measureEveryNthSweep)
		{
			const unsigned int timeT = time0 + stride * measureEveryNthSweep;
			if(timeT >= snapshots.size())
				break;

			OPENMPCD_DEBUG_ASSERT(time0 % measureEveryNthSweep == 0);
			OPENMPCD_DEBUG_ASSERT(timeT % measureEveryNthSweep == 0);
			const Times times(
				time0 / measureEveryNthSweep,
				timeT / measureEveryNthSweep);
			const Values values =
				getAverageNormalCoordinateAutocorrelation(
					chainLength,
					fluid->getNumberOfLogicalEntities(),
					snapshots[time0]->getPointer(),
					snapshots[timeT]->getPointer());

			result.push_back(std::make_pair(times, values));
		}
	}
}




SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"NormalModeAutocorrelation::NormalModeAutocorrelation`",
	"[CUDA]")
{
	using OpenMPCD::CUDA::MPCFluid::Instrumentation::NormalModeAutocorrelation;

	OpenMPCD::CUDA::Simulation simulation(
		OpenMPCDTest::CUDA::SimulationTest::getConfiguration(),
		OpenMPCDTest::CUDA::SimulationTest::getRNGSeed());
	const OpenMPCD::CUDA::MPCFluid::Base* const fluid =
		&simulation.getMPCFluid();

	OpenMPCD::Configuration config =
		OpenMPCDTest::CUDA::SimulationTest::getConfiguration();
	config.set(configGroupName + ".measureEveryNthSweep", 3);
	config.set(configGroupName + ".autocorrelationArgumentCount", 5);


	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			NormalModeAutocorrelation(config, NULL),
			OpenMPCD::NULLPointerException);
	#endif

	{
		const std::vector<OpenMPCD::Configuration> invalidConfigs =
			getInvalidConfigurations();

		for(std::size_t i = 0; i < invalidConfigs.size(); ++i)
		{
			REQUIRE_THROWS_AS(
				NormalModeAutocorrelation(invalidConfigs[i], fluid),
				OpenMPCD::InvalidConfigurationException);
		}
	}

	#ifdef OPENMPCD_DEBUG
		{
			OpenMPCD::RNG rng;
			NonConstantTestFluid testFluid(
				&simulation, 1, simulation.getMPCTimestep(),
				rng, simulation.getDeviceMemoryManager());

			REQUIRE_THROWS_AS(
				NormalModeAutocorrelation(config, &testFluid),
				OpenMPCD::InvalidArgumentException);
		}
	#endif
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"NormalModeAutocorrelation::isConfigured`",
	"[CUDA]")
{
	using OpenMPCD::CUDA::MPCFluid::Instrumentation::NormalModeAutocorrelation;

	OpenMPCD::Configuration config =
		OpenMPCDTest::CUDA::SimulationTest::getConfiguration();

	REQUIRE_FALSE(NormalModeAutocorrelation::isConfigured(config));

	config.createGroup(configGroupName);

	REQUIRE(NormalModeAutocorrelation::isConfigured(config));
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"NormalModeAutocorrelation::isValidConfiguration`",
	"[CUDA]")
{
	using OpenMPCD::CUDA::MPCFluid::Instrumentation::NormalModeAutocorrelation;

	OpenMPCD::Configuration config =
		OpenMPCDTest::CUDA::SimulationTest::getConfiguration();

	config.createGroup(configGroupName);
	REQUIRE_FALSE(
		NormalModeAutocorrelation::isValidConfiguration(
			config.getSetting(configGroupName)));

	{
		const std::vector<OpenMPCD::Configuration> invalidConfigs =
			getInvalidConfigurations();

		for(std::size_t i = 0; i < invalidConfigs.size(); ++i)
		{
			REQUIRE_FALSE(
				NormalModeAutocorrelation::isValidConfiguration(
					invalidConfigs[i].getSetting(configGroupName)));
		}
	}

	config.set(configGroupName + ".measureEveryNthSweep", 3);
	config.set(configGroupName + ".autocorrelationArgumentCount", 5);
	REQUIRE(
		NormalModeAutocorrelation::isValidConfiguration(
			config.getSetting(configGroupName)));

	config.set(configGroupName + ".shift", -0.5);
	REQUIRE(
		NormalModeAutocorrelation::isValidConfiguration(
			config.getSetting(configGroupName)));
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"NormalModeAutocorrelation::getMaximumCorrelationTime`",
	"[CUDA]")
{
	static const unsigned int measureEveryNthSweep = 3;
	static const unsigned int autocorrelationArgumentCount = 5;
	static const unsigned int chainLength = 5;
	static const double shift = 0.123;

	static const unsigned int measurementCount =
		3 * autocorrelationArgumentCount;


	using OpenMPCD::CUDA::MPCFluid::Instrumentation::NormalModeAutocorrelation;

	boost::shared_ptr<OpenMPCD::CUDA::Simulation> simulation;
	boost::shared_ptr<NormalModeAutocorrelation> nma;
	std::vector<
		std::pair<
			std::pair<unsigned int, unsigned int>,
			std::vector<OpenMPCD::MPCParticlePositionType>
			>
		> result;

	getTestcase(
		measureEveryNthSweep, autocorrelationArgumentCount, chainLength,
		measurementCount, shift,
		simulation, nma, result);


	REQUIRE(
		nma->getMaximumCorrelationTime() ==
		autocorrelationArgumentCount - 1);
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"NormalModeAutocorrelation::getMeasurementCount`",
	"[CUDA]")
{
	static const unsigned int measureEveryNthSweep = 3;
	static const unsigned int autocorrelationArgumentCount = 5;
	static const unsigned int chainLength = 5;
	static const double shift = 0.123;

	static const unsigned int measurementCount =
		3 * autocorrelationArgumentCount;


	using OpenMPCD::CUDA::MPCFluid::Instrumentation::NormalModeAutocorrelation;

	boost::shared_ptr<OpenMPCD::CUDA::Simulation> simulation;
	boost::shared_ptr<NormalModeAutocorrelation> nma;
	std::vector<
		std::pair<
			std::pair<unsigned int, unsigned int>,
			std::vector<OpenMPCD::MPCParticlePositionType>
			>
		> result;

	getTestcase(
		measureEveryNthSweep, autocorrelationArgumentCount, chainLength,
		measurementCount, shift,
		simulation, nma, result);



	REQUIRE(nma->getMeasurementCount() == measurementCount);
}


static void test_getAutocorrelation(const double shift)
{
	static const unsigned int measureEveryNthSweep = 3;
	static const unsigned int autocorrelationArgumentCount = 5;
	static const unsigned int chainLength = 5;

	static const unsigned int measurementCount =
		3 * autocorrelationArgumentCount;

	const unsigned int normalModeCount = chainLength + 1;


	using OpenMPCD::CUDA::MPCFluid::Instrumentation::NormalModeAutocorrelation;

	boost::shared_ptr<OpenMPCD::CUDA::Simulation> simulation;
	boost::shared_ptr<NormalModeAutocorrelation> nma;
	std::vector<
		std::pair<
			std::pair<unsigned int, unsigned int>,
			std::vector<OpenMPCD::MPCParticlePositionType>
			>
		> result;

	getTestcase(
		measureEveryNthSweep, autocorrelationArgumentCount, chainLength,
		measurementCount, shift,
		simulation, nma, result);



	unsigned int autocorrelationCount = 0;
	for(unsigned int t = 0; t < measurementCount + 3; ++t)
	{
		for(unsigned int T = 0; T < measurementCount + 3; ++T)
		{
			for(unsigned int n = 0; n < normalModeCount + 3; ++n)
			{
				bool invalid = false;
				if(t >= measurementCount)
					invalid = true;
				if(T >= measurementCount)
					invalid = true;
				if(T < t)
					invalid = true;
				if(T - t > nma->getMaximumCorrelationTime())
					invalid = true;
				if(n >= normalModeCount)
					invalid = true;

				if(invalid)
				{
					#ifdef OPENMPCD_DEBUG
						REQUIRE_THROWS_AS(
							nma->getAutocorrelation(t, T, n),
							OpenMPCD::InvalidArgumentException);
					#endif
					continue;
				}

				const OpenMPCD::MPCParticlePositionType autocorrelation =
					nma->getAutocorrelation(t, T, n);

				bool found = false;
				for(unsigned int r = 0; r < result.size(); ++r)
				{
					const std::pair<unsigned int, unsigned int> times =
						result[r].first;

					if(times.first != t || times.second != T)
						continue;

					REQUIRE(result[r].second.size() == normalModeCount);
					REQUIRE(result[r].second[n] == autocorrelation);

					found = true;
					break;
				}
				REQUIRE(found);

				++autocorrelationCount;
			}
		}
	}

	REQUIRE(result.size() * normalModeCount == autocorrelationCount);
}
SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"NormalModeAutocorrelation::getAutocorrelation`",
	"[CUDA]")
{
	test_getAutocorrelation(std::numeric_limits<double>::quiet_NaN());
	test_getAutocorrelation(0.0);
	test_getAutocorrelation(-0.5);
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"NormalModeAutocorrelation::save(std::ostream&)`",
	"[CUDA]")
{
	static const unsigned int measureEveryNthSweep = 3;
	static const unsigned int autocorrelationArgumentCount = 5;
	static const unsigned int chainLength = 5;
	static const double shift = 0.0;

	static const unsigned int measurementCount =
		3 * autocorrelationArgumentCount;

	const unsigned int normalModeCount = chainLength + 1;


	using OpenMPCD::CUDA::MPCFluid::Instrumentation::NormalModeAutocorrelation;

	boost::shared_ptr<OpenMPCD::CUDA::Simulation> simulation;
	boost::shared_ptr<NormalModeAutocorrelation> nma;
	std::vector<
		std::pair<
			std::pair<unsigned int, unsigned int>,
			std::vector<OpenMPCD::MPCParticlePositionType>
			>
		> result;

	getTestcase(
		measureEveryNthSweep, autocorrelationArgumentCount, chainLength,
		measurementCount, shift,
		simulation, nma, result);



	std::stringstream ss;
	nma->save(ss);

	std::stringstream expected;
	expected.precision(std::numeric_limits<OpenMPCD::FP>::digits10 + 2);

	REQUIRE(nma->getMeasurementCount() > 0);
	for(unsigned int t = 0; t < nma->getMeasurementCount(); ++t)
	{
		for(unsigned int T = t; T <= t + nma->getMaximumCorrelationTime(); ++T)
		{
			if(T >= nma->getMeasurementCount())
				continue;

			expected << t << "\t" << T - t;
			for(unsigned int n = 0; n < normalModeCount; ++n)
				expected << "\t" << nma->getAutocorrelation(t, T, n);
			expected << "\n";
		}
	}

	REQUIRE(expected.str() == ss.str());
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"NormalModeAutocorrelation::save(const std::string&)`",
	"[CUDA]")
{
	static const unsigned int measureEveryNthSweep = 3;
	static const unsigned int autocorrelationArgumentCount = 5;
	static const unsigned int chainLength = 5;
	static const double shift = -0.5;

	static const unsigned int measurementCount =
		3 * autocorrelationArgumentCount;

	const unsigned int normalModeCount = chainLength + 1;


	using OpenMPCD::CUDA::MPCFluid::Instrumentation::NormalModeAutocorrelation;

	boost::shared_ptr<OpenMPCD::CUDA::Simulation> simulation;
	boost::shared_ptr<NormalModeAutocorrelation> nma;
	std::vector<
		std::pair<
			std::pair<unsigned int, unsigned int>,
			std::vector<OpenMPCD::MPCParticlePositionType>
			>
		> result;

	getTestcase(
		measureEveryNthSweep, autocorrelationArgumentCount, chainLength,
		measurementCount, shift,
		simulation, nma, result);



	const std::string rundir =
		std::string("/tmp/") +
		boost::filesystem::unique_path().string() +
		"/rundir";
	const std::string filepath = rundir + "/normalModeAutocorrelations.data";

	SECTION("the rundir does not exist yet")
	{
		REQUIRE_FALSE(boost::filesystem::is_directory(rundir));
	}

	SECTION("the rundir does exist, but does not have the file present")
	{
		REQUIRE_FALSE(boost::filesystem::is_directory(rundir));
		REQUIRE(boost::filesystem::create_directories(rundir));
		REQUIRE(boost::filesystem::is_directory(rundir));
		REQUIRE_FALSE(boost::filesystem::is_regular_file(filepath));
	}

	SECTION("the rundir does exist, and has file present")
	{
		REQUIRE_FALSE(boost::filesystem::is_directory(rundir));
		REQUIRE(boost::filesystem::create_directories(rundir));
		REQUIRE(boost::filesystem::is_directory(rundir));
		REQUIRE_FALSE(boost::filesystem::is_regular_file(filepath));

		std::ofstream file(filepath.c_str(), std::ios::out);
		file << "hello world";
	}


	nma->save(rundir);
	REQUIRE(boost::filesystem::is_directory(rundir));
	REQUIRE(boost::filesystem::is_regular_file(filepath));

	std::ifstream file(filepath.c_str(), std::ios::in);
	std::stringstream filecontents;
	filecontents << file.rdbuf();

	std::stringstream ss;
	nma->save(ss);

	REQUIRE(ss.str() == filecontents.str());

	boost::filesystem::remove_all(rundir);
}
