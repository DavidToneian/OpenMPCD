/**
 * @file
 * Tests
 * `OpenMPCD::CUDA::MPCFluid::Instrumentation::VelocityAutocorrelation`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/VelocityAutocorrelation/Chains.hpp>

#include <OpenMPCDTest/CUDA/SimulationTest.hpp>

#include <OpenMPCD/CUDA/DeviceBuffer.hpp>
#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Chains.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Simple.hpp>
#include <OpenMPCD/CUDA/NormalMode.hpp>

#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>

#include <limits>


static const std::string configGroupName =
	"instrumentation.velocityAutocorrelation";


static
const OpenMPCD::Configuration
getConfig(
	const OpenMPCD::FP measurementTime,
	const unsigned int snapshotCount,
	const unsigned int chainLength)
{
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

	if(measurementTime != 0)
	{
		config.set(
			configGroupName + ".measurementTime",
			measurementTime);
	}

	if(snapshotCount != 0)
	{
		config.set(
			configGroupName + ".snapshotCount",
			snapshotCount);
	}

	return config;
}


static
const std::vector<OpenMPCD::Configuration>
getInvalidConfigurations()
{
	std::vector<OpenMPCD::Configuration> ret;
	const std::string prefix = configGroupName + ".";

	{ //missing `snapshotCount`
		ret.push_back(getConfig(1.0, 0, 5));
	}
	{ //missing `measurementTime`
		ret.push_back(getConfig(0, 2, 5));
	}
	{ //invalid `measurementTime`: smaller than MPC timestep
		ret.push_back(getConfig(0.001, 2, 5));
	}
	{ //invalid `measurementTime`: negative
		ret.push_back(getConfig(-1.0, 2, 5));
	}
	{ //invalid `snapshotCount`: 0
		OpenMPCD::Configuration config = getConfig(1.0, 2, 5);

		config.set(prefix + "snapshotCount", 0);

		ret.push_back(config);
	}

	return ret;
}


static void getTestcase(
	const OpenMPCD::FP measurementTime,
	const unsigned int snapshotCount,
	const unsigned int chainLength,
	const unsigned int sweepCount,
	boost::shared_ptr<OpenMPCD::CUDA::Simulation>& simulation,
	boost::shared_ptr<
		OpenMPCD::CUDA::MPCFluid::Instrumentation::VelocityAutocorrelation::Base
		>& va,
	std::deque<
		boost::tuple<
			OpenMPCD::FP,
			OpenMPCD::FP,
			OpenMPCD::MPCParticleVelocityType>
		>* const V)
{
	const OpenMPCD::Configuration config =
		getConfig(measurementTime, snapshotCount, chainLength);


	simulation.reset(new OpenMPCD::CUDA::Simulation(config, 0));
	const OpenMPCD::CUDA::MPCFluid::Base* const fluid =
		&simulation->getMPCFluid();

	const std::size_t numEntities = fluid->getNumberOfLogicalEntities();


	typedef
		OpenMPCD::CUDA::MPCFluid::
			Instrumentation::VelocityAutocorrelation::Chains
		VAC;
	va.reset(
		new VAC(
			chainLength,
			simulation.get(),
			simulation->getDeviceMemoryManager(),
			fluid));


	using OpenMPCD::CUDA::DeviceBuffer;
	typedef OpenMPCD::MPCParticleVelocityType T;
	typedef
		std::pair<
			OpenMPCD::FP,
			boost::shared_ptr<
				OpenMPCD::CUDA::DeviceBuffer<T>
				>
			>
		Snapshot;

	std::vector<Snapshot> snapshots;
	snapshots.reserve(snapshotCount);


	boost::scoped_array<T> velocities_i(new T[3 * numEntities]);
	boost::scoped_array<T> velocities_j(new T[3 * numEntities]);
	DeviceBuffer<T> deviceBuffer(3 * numEntities);

	using OpenMPCD::CUDA::MPCFluid::DeviceCode::getCenterOfMassVelocities_chain;
	V->clear();
	for(unsigned int m = 0; m < sweepCount; ++m)
	{
		va->measure();

		const OpenMPCD::FP currentTime = simulation->getMPCTime();

		if(snapshots.size() == 0)
		{
			boost::shared_ptr<DeviceBuffer<T> > snapshotBuf(
				new DeviceBuffer<T>(3 * numEntities));
			OpenMPCD::CUDA::DeviceMemoryManager::zeroMemory(
				snapshotBuf->getPointer(),
				3 * numEntities);
			snapshots.push_back(
				std::make_pair(currentTime, snapshotBuf));

			getCenterOfMassVelocities_chain(
				fluid->getParticleCount(),
				chainLength,
				fluid->getDeviceVelocities(),
				snapshotBuf->getPointer());
		}
		else
		{
			if(snapshots.size() < snapshotCount)
			{
				const OpenMPCD::FP deltaT =
					currentTime - snapshots.back().first;

				if(deltaT > measurementTime / snapshotCount - 1e-6)
				{
					boost::shared_ptr<DeviceBuffer<T> > snapshotBuf(
						new DeviceBuffer<T>(3 * numEntities));
					OpenMPCD::CUDA::DeviceMemoryManager::zeroMemory(
						snapshotBuf->getPointer(),
						3 * numEntities);
					snapshots.push_back(
						std::make_pair(currentTime, snapshotBuf));

					getCenterOfMassVelocities_chain(
						fluid->getParticleCount(),
						chainLength,
						fluid->getDeviceVelocities(),
						snapshotBuf->getPointer());
				}
			}
			else
			{
				for(unsigned int i = 0; i < snapshots.size(); ++i)
				{
					const OpenMPCD::FP deltaT =
						currentTime - snapshots[i].first;

					if(deltaT > measurementTime + 1e-6)
					{
						snapshots[i].first = currentTime;

						OpenMPCD::CUDA::DeviceMemoryManager::zeroMemory(
							snapshots[i].second->getPointer(),
							3 * numEntities);

						getCenterOfMassVelocities_chain(
							fluid->getParticleCount(),
							chainLength,
							fluid->getDeviceVelocities(),
							snapshots[i].second->getPointer());
					}
				}
			}
		}


		OpenMPCD::CUDA::MPCFluid::DeviceCode::getCenterOfMassVelocities_chain(
			fluid->getParticleCount(),
			chainLength,
			fluid->getDeviceVelocities(),
			deviceBuffer.getPointer());
		OpenMPCD::CUDA::DeviceMemoryManager::copyElementsFromDeviceToHost(
			deviceBuffer.getPointer(),
			velocities_j.get(),
			3 * numEntities);


		for(unsigned int i = 0; i < snapshots.size(); ++i)
		{
			const OpenMPCD::FP t_i = snapshots[i].first;
			const DeviceBuffer<T>* const snapshot = snapshots[i].second.get();

			REQUIRE(currentTime - t_i >= 0);

			OpenMPCD::CUDA::DeviceMemoryManager::copyElementsFromDeviceToHost(
				snapshot->getPointer(),
				velocities_i.get(),
				3 * numEntities);

			OpenMPCD::FP result = 0;
			for(unsigned int k = 0; k < numEntities; ++k)
			{
				const OpenMPCD::RemotelyStoredVector<T> v_ki(
					velocities_i.get(), k);
				const OpenMPCD::RemotelyStoredVector<T> v_kj(
					velocities_j.get(), k);

				result += v_ki.dot(v_kj);
			}

			result /= numEntities;

			V->push_back(boost::make_tuple(t_i, currentTime, result));
		}

		simulation->sweep();
	}
}




SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"VelocityAutocorrelation::Base`",
	"[CUDA]")
{
	static const unsigned int chainLength = 5;

	typedef
		OpenMPCD::CUDA::MPCFluid::
			Instrumentation::VelocityAutocorrelation::Chains
		VAC;

	const OpenMPCD::Configuration config = getConfig(1.0, 1, chainLength);


	#ifdef OPENMPCD_DEBUG
		{
			OpenMPCD::CUDA::Simulation simulation(
				config,
				OpenMPCDTest::CUDA::SimulationTest::getRNGSeed());
			const OpenMPCD::CUDA::MPCFluid::Base* const fluid =
				&simulation.getMPCFluid();

			REQUIRE_THROWS_AS(
				VAC(
					chainLength,
					NULL,
					simulation.getDeviceMemoryManager(),
					fluid),
				OpenMPCD::NULLPointerException);
			REQUIRE_THROWS_AS(
				VAC(
					chainLength,
					&simulation,
					NULL,
					fluid),
				OpenMPCD::NULLPointerException);
			REQUIRE_THROWS_AS(
				VAC(
					chainLength,
					&simulation,
					simulation.getDeviceMemoryManager(),
					NULL),
				OpenMPCD::NULLPointerException);
		}
	#endif


	{
		const std::vector<OpenMPCD::Configuration> invalidConfigs =
			getInvalidConfigurations();

		for(std::size_t i = 0; i < invalidConfigs.size(); ++i)
		{
			bool caught = false;
			try
			{
				OpenMPCD::CUDA::Simulation simulation(
					invalidConfigs[i],
					OpenMPCDTest::CUDA::SimulationTest::getRNGSeed());
				const OpenMPCD::CUDA::MPCFluid::Base* const fluid =
					&simulation.getMPCFluid();

				VAC(
					chainLength,
					&simulation,
					simulation.getDeviceMemoryManager(),
					fluid);
			}
			catch(const OpenMPCD::InvalidConfigurationException&)
			{
				caught = true;
			}

			REQUIRE(caught);
		}
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"VelocityAutocorrelation::Base::isConfigured`",
	"[CUDA]")
{
	typedef
		OpenMPCD::CUDA::MPCFluid::
			Instrumentation::VelocityAutocorrelation::Chains
		VAC;

	{
		const OpenMPCD::Configuration config =
			OpenMPCDTest::CUDA::SimulationTest::getConfiguration();
		OpenMPCD::CUDA::Simulation simulation(
			config,
			OpenMPCDTest::CUDA::SimulationTest::getRNGSeed());

		REQUIRE_FALSE(VAC::isConfigured(&simulation));

		#ifdef OPENMPCD_DEBUG
			REQUIRE_THROWS_AS(
				VAC::isConfigured(NULL),
				OpenMPCD::NULLPointerException);
		#endif
	}

	{
		const OpenMPCD::Configuration config =
			getConfig(1.0, 1, 5);
		OpenMPCD::CUDA::Simulation simulation(
			config,
			OpenMPCDTest::CUDA::SimulationTest::getRNGSeed());

		REQUIRE(VAC::isConfigured(&simulation));

		#ifdef OPENMPCD_DEBUG
			REQUIRE_THROWS_AS(
				VAC::isConfigured(NULL),
				OpenMPCD::NULLPointerException);
		#endif
	}
}

static void test_save_content(
	const OpenMPCD::FP measurementTime,
	const unsigned int snapshotCount,
	const unsigned int chainLength,
	const unsigned int sweepCount)
{
	using namespace OpenMPCD;

	typedef CUDA::MPCFluid::Instrumentation::VelocityAutocorrelation::Base VAB;
	typedef OpenMPCD::MPCParticleVelocityType T;



	boost::shared_ptr<CUDA::Simulation> simulation;
	boost::shared_ptr<VAB> va;
	std::deque<boost::tuple<FP, FP, T> > V;



	getTestcase(
		measurementTime,
		snapshotCount,
		chainLength,
		sweepCount,
		simulation,
		va,
		&V);



	const std::string rundir =
		std::string("/tmp/") +
		boost::filesystem::unique_path().string() +
		"/rundir";
	const std::string filepath = rundir + "/velocityAutocorrelations.data";

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


	va->save(rundir);

	REQUIRE(boost::filesystem::is_directory(rundir));
	REQUIRE(boost::filesystem::is_regular_file(filepath));

	std::ifstream file(filepath.c_str(), std::ios::in);
	std::stringstream filecontents;
	filecontents << file.rdbuf();

	boost::filesystem::remove_all(rundir);



	for(unsigned int l = 0; l < V.size(); ++l)
	{
		std::string line;
		std::getline(filecontents, line);
		REQUIRE_FALSE(filecontents.eof());

		std::vector<std::string> components;
		boost::algorithm::split(
			components,
			line,
			boost::algorithm::is_any_of("\t"));

		REQUIRE(components.size() == 3);

		//to check for leading/double/trailing tabs:
		const std::string reassembled =
			components[0] + "\t" + components[1] + "\t" + components[2];
		REQUIRE(line == reassembled);

		REQUIRE(boost::lexical_cast<FP>(components[0]) == V[l].get<0>());
		REQUIRE(boost::lexical_cast<FP>(components[1]) == V[l].get<1>());
		REQUIRE(
			boost::lexical_cast<FP>(components[2]) == Approx(V[l].get<2>()));
	}

	{
		std::string line;
		std::getline(filecontents, line);
		REQUIRE(filecontents.eof());
		REQUIRE(line.empty());
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"VelocityAutocorrelation::Base::save(const std::string&)`",
	"[CUDA]")
{
	const OpenMPCD::FP measurementTime = 4;
	const unsigned int snapshotCount = 5;
	const unsigned int chainLength = 5;
	const unsigned int sweepCount = 50;


	test_save_content(measurementTime, snapshotCount, chainLength, sweepCount);
}
