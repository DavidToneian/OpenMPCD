/**
 * @file
 * Tests
 * `OpenMPCD::CUDA::MPCFluid::Instrumentation::LogicalEntityMeanSquareDisplacement`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/LogicalEntityMeanSquareDisplacement.hpp>

#include <OpenMPCDTest/CUDA/SimulationTest.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Simple.hpp>

#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>


static const std::string configGroupName =
	"instrumentation.logicalEntityMeanSquareDisplacement";


using
	OpenMPCD::CUDA::MPCFluid::Instrumentation::
	LogicalEntityMeanSquareDisplacement;

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
		config.set(prefix + "measurementArgumentCount", 3);

		config.set(prefix + "measureEveryNthSweep", 1.0);
		ret.push_back(config);
	}
	{
		OpenMPCD::Configuration config;
		config.set(prefix + "measurementArgumentCount", 3);

		config.set(prefix + "measureEveryNthSweep", 0);
		ret.push_back(config);

		config.set(prefix + "measureEveryNthSweep", -1);
		ret.push_back(config);
	}

	{
		OpenMPCD::Configuration config;

		config.set(prefix + "measurementArgumentCount", 1.0);
		ret.push_back(config);
	}
	{
		OpenMPCD::Configuration config;

		config.set(prefix + "measurementArgumentCount", 0);
		ret.push_back(config);

		config.set(prefix + "measurementArgumentCount", -1);
		ret.push_back(config);
	}
	{
		OpenMPCD::Configuration config;
		config.set(prefix + "measureEveryNthSweep", 5);

		config.set(prefix + "measurementArgumentCount", 1.0);
		ret.push_back(config);
	}
	{
		OpenMPCD::Configuration config;
		config.set(prefix + "measureEveryNthSweep", 5);

		config.set(prefix + "measurementArgumentCount", 0);
		ret.push_back(config);

		config.set(prefix + "measurementArgumentCount", -1);
		ret.push_back(config);
	}

	return ret;
}


static void getTestcase(
	const unsigned int measureEveryNthSweep,
	const unsigned int measurementArgumentCount,
	const unsigned int chainLength,
	const unsigned int measurementCount,
	boost::shared_ptr<OpenMPCD::CUDA::Simulation>& simulation,
	boost::shared_ptr<LogicalEntityMeanSquareDisplacement>& msd,
	std::vector<
		std::pair<
			std::pair<unsigned int, unsigned int>,
			OpenMPCD::MPCParticlePositionType
			>
		>& result)
{
	typedef OpenMPCD::MPCParticlePositionType T;

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
		configGroupName + ".measurementArgumentCount",
		measurementArgumentCount);


	simulation.reset(new OpenMPCD::CUDA::Simulation(config, 0));
	const OpenMPCD::CUDA::MPCFluid::Base* const fluid =
		&simulation->getMPCFluid();


	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	const std::size_t bufferElementCount =
		3 * fluid->getNumberOfLogicalEntities();

	T* buffer;
	dmm.allocateMemory(&buffer, bufferElementCount);

	msd.reset(new LogicalEntityMeanSquareDisplacement(config, fluid));

	std::vector<boost::shared_array<T> > snapshots;

	for(unsigned int m = 0; m < measureEveryNthSweep * measurementCount; ++m)
	{
		msd->measure();

		fluid->saveLogicalEntityCentersOfMassToDeviceMemory(buffer);

		boost::shared_array<T> snapshot(new T[bufferElementCount]);
		snapshots.push_back(snapshot);

		dmm.copyElementsFromDeviceToHost(
			buffer, snapshot.get(), bufferElementCount);

		simulation->sweep();
	}

	typedef std::pair<unsigned int, unsigned int> Times;

	for(
		unsigned int stride = 1;
		stride <= measurementArgumentCount;
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


			T value = 0;
			for(
				unsigned int entity = 0;
				entity < fluid->getNumberOfLogicalEntities();
				++entity)
			{
				const OpenMPCD::RemotelyStoredVector<const T>
					R0(snapshots[time0].get(), entity);
				const OpenMPCD::RemotelyStoredVector<const T>
					RT(snapshots[timeT].get(), entity);

				const OpenMPCD::Vector3D<T> distance = RT - R0;
				value += distance.getMagnitudeSquared();
			}
			value /= fluid->getNumberOfLogicalEntities();

			result.push_back(std::make_pair(times, value));
		}
	}
}




SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"LogicalEntityMeanSquareDisplacement::"
		"LogicalEntityMeanSquareDisplacement`",
	"[CUDA]")
{
	OpenMPCD::CUDA::Simulation simulation(
		OpenMPCDTest::CUDA::SimulationTest::getConfiguration(),
		OpenMPCDTest::CUDA::SimulationTest::getRNGSeed());
	const OpenMPCD::CUDA::MPCFluid::Base* const fluid =
		&simulation.getMPCFluid();

	OpenMPCD::Configuration config =
		OpenMPCDTest::CUDA::SimulationTest::getConfiguration();
	config.set(configGroupName + ".measureEveryNthSweep", 3);
	config.set(configGroupName + ".measurementArgumentCount", 5);


	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			LogicalEntityMeanSquareDisplacement(config, NULL),
			OpenMPCD::NULLPointerException);
	#endif

	{
		const std::vector<OpenMPCD::Configuration> invalidConfigs =
			getInvalidConfigurations();

		for(std::size_t i = 0; i < invalidConfigs.size(); ++i)
		{
			REQUIRE_THROWS_AS(
				LogicalEntityMeanSquareDisplacement(invalidConfigs[i], fluid),
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
				LogicalEntityMeanSquareDisplacement(config, &testFluid),
				OpenMPCD::InvalidArgumentException);
		}
	#endif
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"LogicalEntityMeanSquareDisplacement::isConfigured`",
	"[CUDA]")
{
	OpenMPCD::Configuration config =
		OpenMPCDTest::CUDA::SimulationTest::getConfiguration();

	REQUIRE_FALSE(LogicalEntityMeanSquareDisplacement::isConfigured(config));

	config.createGroup(configGroupName);

	REQUIRE(LogicalEntityMeanSquareDisplacement::isConfigured(config));
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"LogicalEntityMeanSquareDisplacement::isValidConfiguration`",
	"[CUDA]")
{
	OpenMPCD::Configuration config =
		OpenMPCDTest::CUDA::SimulationTest::getConfiguration();

	config.createGroup(configGroupName);
	REQUIRE_FALSE(
		LogicalEntityMeanSquareDisplacement::isValidConfiguration(
			config.getSetting(configGroupName)));

	{
		const std::vector<OpenMPCD::Configuration> invalidConfigs =
			getInvalidConfigurations();

		for(std::size_t i = 0; i < invalidConfigs.size(); ++i)
		{
			REQUIRE_FALSE(
				LogicalEntityMeanSquareDisplacement::isValidConfiguration(
					invalidConfigs[i].getSetting(configGroupName)));
		}
	}

	config.set(configGroupName + ".measureEveryNthSweep", 3);
	config.set(configGroupName + ".measurementArgumentCount", 5);
	REQUIRE(
		LogicalEntityMeanSquareDisplacement::isValidConfiguration(
			config.getSetting(configGroupName)));
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"LogicalEntityMeanSquareDisplacement::getMaximumMeasurementTime`",
	"[CUDA]")
{
	static const unsigned int measureEveryNthSweep = 3;
	static const unsigned int measurementArgumentCount = 5;
	static const unsigned int chainLength = 5;

	static const unsigned int measurementCount =
		3 * measurementArgumentCount;


	boost::shared_ptr<OpenMPCD::CUDA::Simulation> simulation;
	boost::shared_ptr<LogicalEntityMeanSquareDisplacement> msd;
	std::vector<
		std::pair<
			std::pair<unsigned int, unsigned int>,
			OpenMPCD::MPCParticlePositionType
			>
		> result;

	getTestcase(
		measureEveryNthSweep, measurementArgumentCount, chainLength,
		measurementCount,
		simulation, msd, result);


	REQUIRE(msd->getMaximumMeasurementTime() == measurementArgumentCount);
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"LogicalEntityMeanSquareDisplacement::getMeasurementCount`",
	"[CUDA]")
{
	static const unsigned int measureEveryNthSweep = 3;
	static const unsigned int measurementArgumentCount = 5;
	static const unsigned int chainLength = 5;

	static const unsigned int measurementCount =
		3 * measurementArgumentCount;


	boost::shared_ptr<OpenMPCD::CUDA::Simulation> simulation;
	boost::shared_ptr<LogicalEntityMeanSquareDisplacement> msd;
	std::vector<
		std::pair<
			std::pair<unsigned int, unsigned int>,
			OpenMPCD::MPCParticlePositionType
			>
		> result;

	getTestcase(
		measureEveryNthSweep, measurementArgumentCount, chainLength,
		measurementCount,
		simulation, msd, result);



	REQUIRE(msd->getMeasurementCount() == measurementCount);
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"LogicalEntityMeanSquareDisplacement::getMeanSquareDisplacement`",
	"[CUDA]")
{
	static const unsigned int measureEveryNthSweep = 3;
	static const unsigned int measurementArgumentCount = 5;
	static const unsigned int chainLength = 5;

	static const unsigned int measurementCount =
		3 * measurementArgumentCount;


	boost::shared_ptr<OpenMPCD::CUDA::Simulation> simulation;
	boost::shared_ptr<LogicalEntityMeanSquareDisplacement> msd;
	std::vector<
		std::pair<
			std::pair<unsigned int, unsigned int>,
			OpenMPCD::MPCParticlePositionType
			>
		> result;

	getTestcase(
		measureEveryNthSweep, measurementArgumentCount, chainLength,
		measurementCount,
		simulation, msd, result);



	unsigned int measuredMSDCount = 0;
	for(unsigned int t = 0; t < measurementCount + 3; ++t)
	{
		for(unsigned int T = 0; T < measurementCount + 3; ++T)
		{
			bool invalid = false;
			if(t >= measurementCount)
				invalid = true;
			if(T >= measurementCount)
				invalid = true;
			if(T <= t)
				invalid = true;
			if(T - t > msd->getMaximumMeasurementTime())
				invalid = true;

			if(invalid)
			{
				#ifdef OPENMPCD_DEBUG
					REQUIRE_THROWS_AS(
						msd->getMeanSquareDisplacement(t, T),
						OpenMPCD::InvalidArgumentException);
				#endif
				continue;
			}

			const OpenMPCD::MPCParticlePositionType measuredMSD =
				msd->getMeanSquareDisplacement(t, T);

			bool found = false;
			for(unsigned int r = 0; r < result.size(); ++r)
			{
				const std::pair<unsigned int, unsigned int> times =
					result[r].first;

				if(times.first != t || times.second != T)
					continue;

				REQUIRE(result[r].second == measuredMSD);

				found = true;
				break;
			}
			REQUIRE(found);

			++measuredMSDCount;
		}
	}

	REQUIRE(result.size() == measuredMSDCount);
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"LogicalEntityMeanSquareDisplacement::save(std::ostream&)`",
	"[CUDA]")
{
	static const unsigned int measureEveryNthSweep = 3;
	static const unsigned int measurementArgumentCount = 5;
	static const unsigned int chainLength = 5;

	static const unsigned int measurementCount =
		3 * measurementArgumentCount;


	boost::shared_ptr<OpenMPCD::CUDA::Simulation> simulation;
	boost::shared_ptr<LogicalEntityMeanSquareDisplacement> msd;
	std::vector<
		std::pair<
			std::pair<unsigned int, unsigned int>,
			OpenMPCD::MPCParticlePositionType
			>
		> result;

	getTestcase(
		measureEveryNthSweep, measurementArgumentCount, chainLength,
		measurementCount,
		simulation, msd, result);



	std::stringstream ss;
	msd->save(ss);

	std::stringstream expected;
	expected.precision(std::numeric_limits<OpenMPCD::FP>::digits10 + 2);

	REQUIRE(msd->getMeasurementCount() > 0);
	for(unsigned int t = 0; t < msd->getMeasurementCount(); ++t)
	{
		for(
			unsigned int T = t + 1;
			T <= t + msd->getMaximumMeasurementTime();
			++T)
		{
			if(T >= msd->getMeasurementCount())
				continue;

			expected << t << "\t" << T - t;
			expected << "\t" << msd->getMeanSquareDisplacement(t, T);
			expected << "\n";
		}
	}

	REQUIRE(expected.str() == ss.str());
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"LogicalEntityMeanSquareDisplacement::save(const std::string&)`",
	"[CUDA]")
{
	static const unsigned int measureEveryNthSweep = 3;
	static const unsigned int measurementArgumentCount = 5;
	static const unsigned int chainLength = 5;

	static const unsigned int measurementCount =
		3 * measurementArgumentCount;


	boost::shared_ptr<OpenMPCD::CUDA::Simulation> simulation;
	boost::shared_ptr<LogicalEntityMeanSquareDisplacement> msd;
	std::vector<
		std::pair<
			std::pair<unsigned int, unsigned int>,
			OpenMPCD::MPCParticlePositionType
			>
		> result;

	getTestcase(
		measureEveryNthSweep, measurementArgumentCount, chainLength,
		measurementCount,
		simulation, msd, result);



	const std::string rundir =
		std::string("/tmp/") +
		boost::filesystem::unique_path().string() +
		"/rundir";
	const std::string filepath =
		rundir + "/logicalEntityMeanSquareDisplacement.data";

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


	msd->save(rundir);
	REQUIRE(boost::filesystem::is_directory(rundir));
	REQUIRE(boost::filesystem::is_regular_file(filepath));

	std::ifstream file(filepath.c_str(), std::ios::in);
	std::stringstream filecontents;
	filecontents << file.rdbuf();

	std::stringstream ss;
	msd->save(ss);

	REQUIRE(ss.str() == filecontents.str());

	boost::filesystem::remove_all(rundir);
}
