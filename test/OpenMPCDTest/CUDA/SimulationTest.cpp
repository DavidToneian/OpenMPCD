/**
 * @file
 * Tests `OpenMPCD::CUDA::Simulation`.
 */

#include <OpenMPCDTest/CUDA/SimulationTest.hpp>

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/MPCFluid/Simple.hpp>

#include <boost/filesystem.hpp>
#include <boost/scoped_ptr.hpp>

typedef void (*CreateInstanceFunction)(
	boost::scoped_ptr<OpenMPCD::CUDA::Simulation>* const,
	const OpenMPCD::Configuration&);

static void createInstance_2args(
	boost::scoped_ptr<OpenMPCD::CUDA::Simulation>* const ptr,
	const OpenMPCD::Configuration& config)
{
	ptr->reset(new OpenMPCD::CUDA::Simulation(config, 0));
}

static void createInstance_3args(
	boost::scoped_ptr<OpenMPCD::CUDA::Simulation>* const ptr,
	const OpenMPCD::Configuration& config)
{
	const boost::filesystem::path tempdir =
		boost::filesystem::temp_directory_path() /
		boost::filesystem::unique_path();
	boost::filesystem::create_directories(tempdir);

	const boost::filesystem::path configPath = tempdir / "config.txt";
	const boost::filesystem::path rundir = tempdir / "rundir";

	config.writeToFile(configPath.c_str());
	boost::filesystem::create_directories(rundir);

	ptr->reset(
		new OpenMPCD::CUDA::Simulation(configPath.c_str(), 0, tempdir.c_str()));
}


static void getNumberOfCompletedSweeps_test(
		const CreateInstanceFunction createInstance)
{
	typedef OpenMPCD::CUDA::Simulation Simulation;

	GIVEN("`Simulation` instances with various warmup/sweep configurations")
	{
		OpenMPCD::Configuration config =
			OpenMPCDTest::CUDA::SimulationTest::getConfiguration();

		boost::scoped_ptr<Simulation> sim[4];

		config.set("mpc.warmupSteps", 0);
		config.set("mpc.sweepSize", 1);
		createInstance(&sim[0], config);

		config.set("mpc.warmupSteps", 10);
		createInstance(&sim[1], config);

		config.set("mpc.sweepSize", 3);
		createInstance(&sim[2], config);

		config.set("mpc.timestep", 2.0);
		createInstance(&sim[3], config);

		THEN("`getNumberOfCompletedSweeps` returns correct values")
		{
			for(std::size_t i = 0; i < sizeof(sim)/sizeof(sim[0]); ++i)
			{
				REQUIRE(sim[i]->getNumberOfCompletedSweeps() == 0);

				sim[i]->warmup();

				REQUIRE(sim[i]->getNumberOfCompletedSweeps() == 0);

				sim[i]->sweep();

				REQUIRE(sim[i]->getNumberOfCompletedSweeps() == 1);

				sim[i]->sweep();
				sim[i]->sweep();
				REQUIRE(sim[i]->getNumberOfCompletedSweeps() == 3);
			}
		}
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::Simulation::getNumberOfCompletedSweeps`",
	"[CUDA]")
{
	const CreateInstanceFunction createInstanceFunctions[] =
		{
			createInstance_2args,
			createInstance_3args
		};

	for(
		std::size_t i = 0;
		i < sizeof(createInstanceFunctions)/sizeof(createInstanceFunctions[0]);
		++i)
	{
		SECTION("", "", i)
		{
			getNumberOfCompletedSweeps_test(createInstanceFunctions[i]);
		}
	}
}


static void getSimulationBoxSize_XYZ_test(
	const CreateInstanceFunction createInstance)
{
	typedef OpenMPCD::CUDA::Simulation Simulation;

	static const unsigned int boxSizes[] = {1, 2, 5};
	static const std::size_t boxSizeCount =
		sizeof(boxSizes)/sizeof(boxSizes[0]);


	for(std::size_t BSXIndex = 0; BSXIndex < boxSizeCount; ++BSXIndex)
	{
		const unsigned int boxSizeX = boxSizes[BSXIndex];

		for(std::size_t BSYIndex = 0; BSYIndex < boxSizeCount; ++BSYIndex)
		{
			const unsigned int boxSizeY = boxSizes[BSYIndex];

			for(std::size_t BSZIndex = 0; BSZIndex < boxSizeCount; ++BSZIndex)
			{
				const unsigned int boxSizeZ = boxSizes[BSZIndex];

				OpenMPCD::Configuration config =
					OpenMPCDTest::CUDA::SimulationTest::getConfiguration();

				config.set("mpc.simulationBoxSize.x", boxSizeX);
				config.set("mpc.simulationBoxSize.y", boxSizeY);
				config.set("mpc.simulationBoxSize.z", boxSizeZ);

				boost::scoped_ptr<Simulation> sim;
				createInstance(&sim, config);

				REQUIRE(sim->getSimulationBoxSizeX() == boxSizeX);
				REQUIRE(sim->getSimulationBoxSizeY() == boxSizeY);
				REQUIRE(sim->getSimulationBoxSizeZ() == boxSizeZ);

				sim->sweep();

				REQUIRE(sim->getSimulationBoxSizeX() == boxSizeX);
				REQUIRE(sim->getSimulationBoxSizeY() == boxSizeY);
				REQUIRE(sim->getSimulationBoxSizeZ() == boxSizeZ);
			}
		}
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::Simulation::getSimulationBoxSizeX`, "
		"`OpenMPCD::CUDA::Simulation::getSimulationBoxSizeY`, "
		"`OpenMPCD::CUDA::Simulation::getSimulationBoxSizeZ`",
	"[CUDA]")
{
	const CreateInstanceFunction createInstanceFunctions[] =
		{
			createInstance_2args,
			createInstance_3args
		};

	for(
		std::size_t i = 0;
		i < sizeof(createInstanceFunctions)/sizeof(createInstanceFunctions[0]);
		++i)
	{
		SECTION("", "", i)
		{
			getSimulationBoxSize_XYZ_test(createInstanceFunctions[i]);
		}
	}
}


static void getCollisionCellCount_test(
	const CreateInstanceFunction createInstance)
{
	typedef OpenMPCD::CUDA::Simulation Simulation;

	static const unsigned int boxSizes[] = {2, 5};
	static const std::size_t boxSizeCount =
		sizeof(boxSizes)/sizeof(boxSizes[0]);


	for(std::size_t BSXIndex = 0; BSXIndex < boxSizeCount; ++BSXIndex)
	{
		const unsigned int boxSizeX = boxSizes[BSXIndex];

		for(std::size_t BSYIndex = 0; BSYIndex < boxSizeCount; ++BSYIndex)
		{
			const unsigned int boxSizeY = boxSizes[BSYIndex];

			for(std::size_t BSZIndex = 0; BSZIndex < boxSizeCount; ++BSZIndex)
			{
				const unsigned int boxSizeZ = boxSizes[BSZIndex];

				OpenMPCD::Configuration config =
					OpenMPCDTest::CUDA::SimulationTest::getConfiguration();

				config.set("mpc.simulationBoxSize.x", boxSizeX);
				config.set("mpc.simulationBoxSize.y", boxSizeY);
				config.set("mpc.simulationBoxSize.z", boxSizeZ);

				boost::scoped_ptr<Simulation> sim;
				createInstance(&sim, config);


				const unsigned int expected = boxSizeX * boxSizeY * boxSizeZ;

				REQUIRE(sim->getCollisionCellCount() == expected);

				sim->sweep();

				REQUIRE(sim->getCollisionCellCount() == expected);
			}
		}
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::Simulation::getCollisionCellCount`",
	"[CUDA]")
{
	const CreateInstanceFunction createInstanceFunctions[] =
		{
			createInstance_2args,
			createInstance_3args
		};

	for(
		std::size_t i = 0;
		i < sizeof(createInstanceFunctions)/sizeof(createInstanceFunctions[0]);
		++i)
	{
		SECTION("", "", i)
		{
			getCollisionCellCount_test(createInstanceFunctions[i]);
		}
	}
}


static void hasMPCFluid_test(const CreateInstanceFunction createInstance)
{
	using OpenMPCDTest::CUDA::SimulationTest;

	boost::scoped_ptr<OpenMPCD::CUDA::Simulation> sim;

	GIVEN("a Simulation instance without an MPC fluid configured")
	{
		createInstance(&sim, SimulationTest::getConfigurationWithoutFluid());

		THEN("`hasMPCFluid` returns `false`")
		{
			REQUIRE_FALSE(sim->hasMPCFluid());
		}
	}

	GIVEN("a Simulation instance with an MPC fluid configured")
	{
		createInstance(&sim, SimulationTest::getConfiguration());

		THEN("`hasMPCFluid` returns `true`")
		{
			REQUIRE(sim->hasMPCFluid());
		}
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::Simulation::hasMPCFluid`",
	"[CUDA]")
{
	const CreateInstanceFunction createInstanceFunctions[] =
		{
			createInstance_2args,
			createInstance_3args
		};

	for(
		std::size_t i = 0;
		i < sizeof(createInstanceFunctions)/sizeof(createInstanceFunctions[0]);
		++i)
	{
		SECTION("", "", i)
		{
			hasMPCFluid_test(createInstanceFunctions[i]);
		}
	}
}


static void getMPCFluid_test(const CreateInstanceFunction createInstance)
{
	using OpenMPCDTest::CUDA::SimulationTest;

	boost::scoped_ptr<OpenMPCD::CUDA::Simulation> sim;

	GIVEN("a Simulation instance without an MPC fluid configured")
	{
		createInstance(&sim, SimulationTest::getConfigurationWithoutFluid());

		THEN("`getMPCFluid` throws if `OPENMPCD_DEBUG` is defined")
		{
			#ifdef OPENMPCD_DEBUG
				REQUIRE_THROWS_AS(
					sim->getMPCFluid(),
					OpenMPCD::InvalidCallException);
			#endif
		}
	}

	GIVEN("a Simulation instance with an MPC fluid configured")
	{
		const OpenMPCD::Configuration config =
			OpenMPCDTest::CUDA::SimulationTest::getConfiguration();

		createInstance(&sim, config);

		THEN("`getMPCFluid` works")
		{
			REQUIRE_NOTHROW(sim->getMPCFluid());

			REQUIRE(
				dynamic_cast<const OpenMPCD::CUDA::MPCFluid::Simple*>(
					&sim->getMPCFluid())
				!= NULL);

			const unsigned int expectedParticleCount =
				config.read<unsigned int>("initialization.particleDensity") *
				config.read<unsigned int>("mpc.simulationBoxSize.x") *
				config.read<unsigned int>("mpc.simulationBoxSize.y") *
				config.read<unsigned int>("mpc.simulationBoxSize.z");
			REQUIRE(
				sim->getMPCFluid().getParticleCount() == expectedParticleCount);
		}
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::Simulation::getMPCFluid`",
	"[CUDA]")
{
	const CreateInstanceFunction createInstanceFunctions[] =
		{
			createInstance_2args,
			createInstance_3args
		};

	for(
		std::size_t i = 0;
		i < sizeof(createInstanceFunctions)/sizeof(createInstanceFunctions[0]);
		++i)
	{
		SECTION("", "", i)
		{
			getMPCFluid_test(createInstanceFunctions[i]);
		}
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::Simulation::generateCollisionCellMBSFactors`",
	"[CUDA]")
{
	OpenMPCDTest::CUDA::SimulationTest::test_generateCollisionCellMBSFactors();
}
void OpenMPCDTest::CUDA::SimulationTest::test_generateCollisionCellMBSFactors()
{
	GIVEN("a collision cell with 0 particles in it")
	{
		OpenMPCD::CUDA::Simulation simulation(getConfiguration(), getRNGSeed());
		WARN("test incomplete");
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::Simulation::stream` with simple fluid",
	"[CUDA]")
{
	OpenMPCDTest::CUDA::SimulationTest::test_stream_simpleFluid();
}
void OpenMPCDTest::CUDA::SimulationTest::test_stream_simpleFluid()
{
	static const unsigned int stepCount = 5;

	using namespace OpenMPCD;
	using namespace OpenMPCD::CUDA;

	Simulation simulation(getConfiguration(), getRNGSeed());

	const MPCFluid::Base& fluid = simulation.getMPCFluid();
	fluid.fetchFromDevice();

	MPCParticlePositionType* const oldPositions =
		new MPCParticlePositionType[fluid.getParticleCount() * 3];
	MPCParticleVelocityType* const oldVelocities =
		new MPCParticleVelocityType[fluid.getParticleCount() * 3];


	//to make sure the test isn't trivial:
	bool foundNonZeroVelocity[3];
	foundNonZeroVelocity[0] = false;
	foundNonZeroVelocity[1] = false;
	foundNonZeroVelocity[2] = false;

	for(unsigned int i = 0; i < fluid.getParticleCount(); ++i)
	{
		RemotelyStoredVector<MPCParticlePositionType> pos(oldPositions, i);
		RemotelyStoredVector<MPCParticleVelocityType> vel(oldVelocities, i);

		pos = fluid.getPosition(i);
		vel = fluid.getVelocity(i);

		if(vel.getX() != 0)
			foundNonZeroVelocity[0] = true;
		if(vel.getY() != 0)
			foundNonZeroVelocity[1] = true;
		if(vel.getZ() != 0)
			foundNonZeroVelocity[2] = true;
	}

	REQUIRE(foundNonZeroVelocity[0]);
	REQUIRE(foundNonZeroVelocity[1]);
	REQUIRE(foundNonZeroVelocity[2]);

	for(unsigned int step = 0; step < stepCount; ++step)
	{
		simulation.stream();
		fluid.fetchFromDevice();

		for(unsigned int i = 0; i < fluid.getParticleCount(); ++i)
		{
			RemotelyStoredVector<MPCParticlePositionType> pos(oldPositions, i);
			RemotelyStoredVector<MPCParticleVelocityType> vel(oldVelocities, i);

			REQUIRE(
				fluid.getPosition(i)
				==
				pos + vel * simulation.getMPCTimestep());

			REQUIRE(fluid.getVelocity(i) == vel);

			pos = fluid.getPosition(i);
		}
	}

	delete[] oldPositions;
	delete[] oldVelocities;
}



const OpenMPCD::Configuration OpenMPCDTest::CUDA::SimulationTest::getConfiguration()
{
	OpenMPCD::Configuration config;
	config.set("initialization.particleDensity", 10);
	config.set("initialization.particleVelocityDistribution.mean", 0.0);
	config.set("initialization.particleVelocityDistribution.standardDeviation", 1.0);

	config.set("mpc.simulationBoxSize.x", 10);
	config.set("mpc.simulationBoxSize.y", 10);
	config.set("mpc.simulationBoxSize.z", 10);
	config.set("mpc.timestep",            0.1);
	config.set("mpc.srdCollisionAngle",   2.27);
	config.set("mpc.gridShiftScale",      1.0);

	config.set("bulkThermostat.targetkT", 1.0);

	config.set("mpc.sweepSize", 1);

	config.set("bulkThermostat.type", "MBS");

	config.set("boundaryConditions.LeesEdwards.shearRate", 0.0);

	config.createGroup("mpc.fluid.simple");

	return config;
}

const OpenMPCD::Configuration
OpenMPCDTest::CUDA::SimulationTest::getConfigurationWithoutFluid()
{
	OpenMPCD::Configuration config;
	config.set("initialization.particleDensity", 0);

	config.set("mpc.simulationBoxSize.x", 10);
	config.set("mpc.simulationBoxSize.y", 10);
	config.set("mpc.simulationBoxSize.z", 10);
	config.set("mpc.timestep",            0.1);
	config.set("mpc.srdCollisionAngle",   2.27);
	config.set("mpc.gridShiftScale",      1.0);

	config.set("bulkThermostat.targetkT", 1.0);

	config.set("mpc.sweepSize", 1);

	config.set("bulkThermostat.type", "MBS");

	config.set("boundaryConditions.LeesEdwards.shearRate", 0.0);

	return config;
}

unsigned int OpenMPCDTest::CUDA::SimulationTest::getRNGSeed()
{
	return 12345;
}
