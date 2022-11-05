/**
 * @file
 * Tests `OpenMPCD::CUDA::MPCFluid::Base`.
 */

#include <OpenMPCDTest/CUDA/MPCFluid/BaseTest.hpp>

#include <OpenMPCDTest/CUDA/SimulationTest.hpp>
#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Base.hpp>

#include <boost/filesystem/operations.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/scoped_array.hpp>


OpenMPCDTest::CUDA::MPCFluid::BaseTest::BaseTest(
	const OpenMPCD::CUDA::Simulation* const sim, const unsigned int count, const OpenMPCD::FP streamingTimestep_,
	OpenMPCD::RNG& rng_, OpenMPCD::CUDA::DeviceMemoryManager* const devMemMgr)
		: Base(sim, count, streamingTimestep_, rng_, devMemMgr)
{
	particlesPerLogicalEntity = 1;
}

SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Base::getParticleCount`",
	"[CUDA]")
{
	OpenMPCDTest::CUDA::MPCFluid::BaseTest::executeWithVariousDensitiesAndGeometries(
		&OpenMPCDTest::CUDA::MPCFluid::BaseTest::test_getParticleCount);
}
void OpenMPCDTest::CUDA::MPCFluid::BaseTest::test_getParticleCount(const unsigned int particlesPerCell, const unsigned int (&simulationBox)[3])
{
	GIVEN("a simulation with a certain density and geometry")
	{
		const std::pair<boost::shared_ptr<OpenMPCD::CUDA::Simulation>, boost::shared_ptr<BaseTest> >
			simAndFluid = getSimulationAndFluid(particlesPerCell, simulationBox);

		BaseTest* const fluid = simAndFluid.second.get();

		THEN("the particle count is correct")
		{
			const unsigned int particleCount = particlesPerCell * simulationBox[0] * simulationBox[1] * simulationBox[2];
			REQUIRE(fluid->getParticleCount() == particleCount);
		}
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Base::getParticleMass`",
	"[CUDA]")
{
	REQUIRE(OpenMPCD::CUDA::MPCFluid::Base::getParticleMass() == 1);
}


SCENARIO(
	"Certain uses of `fetchFromDevice` and `pushToDevice` of `OpenMPCD::CUDA::MPCFluid::Base` do not alter the state",
	"[CUDA]")
{
	OpenMPCDTest::CUDA::MPCFluid::BaseTest::executeWithVariousDensitiesAndGeometries(
		&OpenMPCDTest::CUDA::MPCFluid::BaseTest::testFetchAndPushInvariants);
}
void OpenMPCDTest::CUDA::MPCFluid::BaseTest::testFetchAndPushInvariants(
	const unsigned int particlesPerCell, const unsigned int (&simulationBox)[3])
{
	GIVEN("a simulation with a certain density and geometry")
	{
		const std::pair<boost::shared_ptr<OpenMPCD::CUDA::Simulation>, boost::shared_ptr<BaseTest> >
			simAndFluid = getSimulationAndFluid(particlesPerCell, simulationBox);

		BaseTest* const fluid = simAndFluid.second.get();


		const std::size_t coordinateCount = 3 * fluid->getParticleCount();

		boost::scoped_array<OpenMPCD::MPCParticlePositionType> positionBuffer(
			new OpenMPCD::MPCParticlePositionType[coordinateCount]);
		boost::scoped_array<OpenMPCD::MPCParticleVelocityType> velocityBuffer(
			new OpenMPCD::MPCParticleVelocityType[coordinateCount]);

		const std::size_t positionBufferByteCount = coordinateCount * sizeof(positionBuffer[0]);
		const std::size_t velocityBufferByteCount = coordinateCount * sizeof(velocityBuffer[0]);


		memcpy(positionBuffer.get(), fluid->getHostPositions(), positionBufferByteCount);
		memcpy(velocityBuffer.get(), fluid->getHostVelocities(), velocityBufferByteCount);

		WHEN("`pushToDevice` is called")
		{
			fluid->pushToDevice();

			THEN("the state on the Host is unchanged")
			{
				REQUIRE(memcmp(positionBuffer.get(), fluid->getHostPositions(), positionBufferByteCount) == 0);
				REQUIRE(memcmp(velocityBuffer.get(), fluid->getHostVelocities(), velocityBufferByteCount) == 0);
			}
		}

		WHEN("`pushToDevice` and then `fetchFromDevice` are called")
		{
			fluid->pushToDevice();
			fluid->fetchFromDevice();

			THEN("the state on the Host is unchanged")
			{
				REQUIRE(memcmp(positionBuffer.get(), fluid->getHostPositions(), positionBufferByteCount) == 0);
				REQUIRE(memcmp(velocityBuffer.get(), fluid->getHostVelocities(), velocityBufferByteCount) == 0);
			}
		}
	}
}



SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Base::setPositionsAndVelocities`",
	"[CUDA]")
{
	OpenMPCDTest::CUDA::MPCFluid::BaseTest::executeWithVariousDensitiesAndGeometries(
		&OpenMPCDTest::CUDA::MPCFluid::BaseTest::test_setPositionsAndVelocities);
}
void OpenMPCDTest::CUDA::MPCFluid::BaseTest::test_setPositionsAndVelocities(
	const unsigned int particlesPerCell, const unsigned int (&simulationBox)[3])
{
	GIVEN("a simulation with a certain density and geometry")
	{
		const std::pair<boost::shared_ptr<OpenMPCD::CUDA::Simulation>, boost::shared_ptr<BaseTest> >
			simAndFluid = getSimulationAndFluid(particlesPerCell, simulationBox);

		BaseTest* const fluid = simAndFluid.second.get();


		const std::size_t coordinateCount = 3 * fluid->getParticleCount();

		boost::scoped_array<OpenMPCD::MPCParticlePositionType> positionBuffer(
			new OpenMPCD::MPCParticlePositionType[coordinateCount]);
		boost::scoped_array<OpenMPCD::MPCParticleVelocityType> velocityBuffer(
			new OpenMPCD::MPCParticleVelocityType[coordinateCount]);

		const std::size_t positionBufferByteCount = coordinateCount * sizeof(positionBuffer[0]);
		const std::size_t velocityBufferByteCount = coordinateCount * sizeof(velocityBuffer[0]);

		for(std::size_t coord = 0; coord < coordinateCount; ++coord)
		{
			positionBuffer[coord] = coord;
			velocityBuffer[coord] = 1.0 / coord;
		}

		WHEN("`setPositionsAndVelocities` is called")
		{
			fluid->setPositionsAndVelocities(
				positionBuffer.get(), velocityBuffer.get());

			THEN("the positions and velocities on the Device are correct")
			{
				fluid->fetchFromDevice();
				REQUIRE(
					memcmp(
						positionBuffer.get(),
						fluid->getHostPositions(),
						positionBufferByteCount)
					== 0);
				REQUIRE(
					memcmp(
						velocityBuffer.get(),
						fluid->getHostVelocities(),
						velocityBufferByteCount)
					== 0);
			}
		}
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Base::writeToSnapshot`",
	"[CUDA]")
{
	const unsigned int maxParticleCount = 30 * 10 * 10 * 10;

	OpenMPCDTest::CUDA::MPCFluid::BaseTest::executeWithVariousDensitiesAndGeometries(
		&OpenMPCDTest::CUDA::MPCFluid::BaseTest::test_writeToSnapshot,
		maxParticleCount);
}
void OpenMPCDTest::CUDA::MPCFluid::BaseTest::test_writeToSnapshot(
	const unsigned int particlesPerCell, const unsigned int (&simulationBox)[3])
{
	using namespace OpenMPCD;

	static const char* const path = "test/data/VTFSnapshotFile/tmp.vtf";

	GIVEN("a simulation with a certain density and geometry")
	{
		const std::pair<boost::shared_ptr<OpenMPCD::CUDA::Simulation>, boost::shared_ptr<BaseTest> >
			simAndFluid = getSimulationAndFluid(particlesPerCell, simulationBox);

		BaseTest* const fluid = simAndFluid.second.get();

		const std::size_t coordinateCount = 3 * fluid->getParticleCount();

		for(std::size_t coord = 0; coord < coordinateCount; ++coord)
		{
			fluid->getHostPositions()[coord] = coord;
			fluid->getHostVelocities()[coord] = 1.0 / coord;
		}

		fluid->pushToDevice();

		THEN("passing `nullptr` throws")
		{
			REQUIRE_THROWS_AS(
				fluid->writeToSnapshot(NULL),
				OpenMPCD::NULLPointerException);
		}

		THEN("not setting atom count throws")
		{
			boost::filesystem::remove(path);
			REQUIRE_FALSE(boost::filesystem::is_regular_file(path));

			VTFSnapshotFile snapshot(path);
			REQUIRE(snapshot.isInWriteMode());

			REQUIRE_THROWS_AS(
				fluid->writeToSnapshot(&snapshot),
				OpenMPCD::InvalidArgumentException);
		}

		THEN("setting atom count incorrectly throws")
		{
			boost::filesystem::remove(path);
			REQUIRE_FALSE(boost::filesystem::is_regular_file(path));

			VTFSnapshotFile snapshot(path);
			snapshot.declareAtoms(fluid->getParticleCount() + 1);
			REQUIRE(snapshot.isInWriteMode());

			REQUIRE_THROWS_AS(
				fluid->writeToSnapshot(&snapshot),
				OpenMPCD::InvalidArgumentException);
		}

		THEN("throws on read mode")
		{
			boost::filesystem::remove(path);
			REQUIRE_FALSE(boost::filesystem::is_regular_file(path));
			{
				VTFSnapshotFile snapshot(path);
				snapshot.declareAtoms(fluid->getParticleCount());
			}
			REQUIRE(boost::filesystem::is_regular_file(path));

			VTFSnapshotFile snapshot(path);
			REQUIRE(snapshot.getNumberOfAtoms() == fluid->getParticleCount());
			REQUIRE_FALSE(snapshot.isInWriteMode());

			REQUIRE_THROWS_AS(
				fluid->writeToSnapshot(&snapshot),
				OpenMPCD::InvalidArgumentException);
		}

		THEN("`writeToSnapshot` works")
		{
			{
				boost::filesystem::remove(path);
				REQUIRE_FALSE(boost::filesystem::is_regular_file(path));

				VTFSnapshotFile snapshot(path);
				REQUIRE(snapshot.isInWriteMode());

				snapshot.declareAtoms(fluid->getParticleCount());

				fluid->writeToSnapshot(&snapshot);
			}

			{
				VTFSnapshotFile snapshot(path);
				REQUIRE_FALSE(snapshot.isInWriteMode());
				REQUIRE(snapshot.getNumberOfAtoms() == fluid->getParticleCount());

				boost::scoped_array<MPCParticlePositionType>
					positionReadBuffer(
							new MPCParticlePositionType[coordinateCount]);
				boost::scoped_array<OpenMPCD::MPCParticleVelocityType>
					velocityReadBuffer(
							new MPCParticleVelocityType[coordinateCount]);
				bool velocitiesEncountered = false;

				snapshot.readTimestepBlock(
					positionReadBuffer.get(),
					velocityReadBuffer.get(),
					&velocitiesEncountered);

				REQUIRE(velocitiesEncountered);

				for(std::size_t coord = 0; coord < coordinateCount; ++coord)
				{
					REQUIRE(positionReadBuffer[coord] == coord);
					REQUIRE(velocityReadBuffer[coord] == 1.0 / coord);
				}
			}

			boost::filesystem::remove(path);
			REQUIRE_FALSE(boost::filesystem::is_regular_file(path));
		}
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Base::findMatchingParticlesOnHost`",
	"[CUDA]")
{
	OpenMPCDTest::CUDA::MPCFluid::BaseTest::executeWithVariousDensitiesAndGeometries(
		&OpenMPCDTest::CUDA::MPCFluid::BaseTest::test_findMatchingParticlesOnHost);
}
void OpenMPCDTest::CUDA::MPCFluid::BaseTest::test_findMatchingParticlesOnHost(
	const unsigned int particlesPerCell, const unsigned int (&simulationBox)[3])
{
	using namespace OpenMPCD;

	GIVEN("a simulation with a certain density and geometry")
	{
		const std::pair<boost::shared_ptr<OpenMPCD::CUDA::Simulation>, boost::shared_ptr<BaseTest> >
			simAndFluid = getSimulationAndFluid(particlesPerCell, simulationBox);

		BaseTest* const fluid = simAndFluid.second.get();
		OpenMPCD::RNG* const rng = SimulationTest::getRNG(simAndFluid.first.get());


		//initialize Host state
		std::vector<unsigned int> expectedMatches;
		expectedMatches.reserve(fluid->getParticleCount());
		boost::random::uniform_01<MPCParticlePositionType> positionDist;
		boost::random::uniform_01<MPCParticleVelocityType> velocityDist;
		for(unsigned int i=0; i<fluid->getParticleCount(); ++i)
		{
			fluid->getHostPositions()[3 * i + 0] = simulationBox[0] * positionDist(*rng);
			fluid->getHostPositions()[3 * i + 1] = simulationBox[1] * positionDist(*rng);
			fluid->getHostPositions()[3 * i + 2] = simulationBox[2] * positionDist(*rng);

			fluid->getHostVelocities()[3 * i + 0] = velocityDist(*rng);
			fluid->getHostVelocities()[3 * i + 1] = velocityDist(*rng);
			fluid->getHostVelocities()[3 * i + 2] = velocityDist(*rng);

			if(fluid->getHostVelocities()[3 * i + 2] < 0.5)
				expectedMatches.push_back(i);
		}

		THEN("`findMatchingParticlesOnHost` returns the expected result")
		{
			class Matcher
			{
				public:
				static bool match(
					const RemotelyStoredVector<const MPCParticlePositionType>&,
					const RemotelyStoredVector<const MPCParticleVelocityType>& v)
				{
					return v.getZ() < 0.5;
				}
			};

			REQUIRE_NOTHROW(fluid->findMatchingParticlesOnHost(&Matcher::match, NULL, NULL));

			{
				std::vector<unsigned int> matches;
				REQUIRE_NOTHROW(fluid->findMatchingParticlesOnHost(&Matcher::match, &matches, NULL));
				REQUIRE(matches == expectedMatches);
			}

			{
				unsigned int matchCount = 12345;
				REQUIRE_NOTHROW(fluid->findMatchingParticlesOnHost(&Matcher::match, NULL, &matchCount));
				REQUIRE(matchCount == expectedMatches.size());
			}

			{
				std::vector<unsigned int> matches;
				unsigned int matchCount;
				REQUIRE_NOTHROW(fluid->findMatchingParticlesOnHost(&Matcher::match, &matches, &matchCount));
				REQUIRE(matches == expectedMatches);
				REQUIRE(matchCount == expectedMatches.size());
			}
		}
	}
}



SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Base::"
		"saveLogicalEntityCentersOfMassToDeviceMemory`",
	"[CUDA]")
{
	using OpenMPCDTest::CUDA::MPCFluid::BaseTest;
	BaseTest::executeWithVariousDensitiesAndGeometries(
		&BaseTest::test_saveLogicalEntityCentersOfMassToDeviceMemory,
		10 * 10 * 10 * 10);
}
void
OpenMPCDTest::CUDA::MPCFluid::BaseTest::
	test_saveLogicalEntityCentersOfMassToDeviceMemory(
		const unsigned int particlesPerCell,
		const unsigned int (&simulationBox)[3])
{
	using namespace OpenMPCD;

	GIVEN("a simulation with a certain density and geometry")
	{
		const std::pair<
			boost::shared_ptr<OpenMPCD::CUDA::Simulation>,
			boost::shared_ptr<BaseTest> >
				simAndFluid =
					getSimulationAndFluid(particlesPerCell, simulationBox);

		BaseTest* const fluid = simAndFluid.second.get();
		OpenMPCD::RNG* const rng =
			SimulationTest::getRNG(simAndFluid.first.get());


		//initialize Host state
		boost::random::uniform_01<MPCParticlePositionType> positionDist;
		for(unsigned int i=0; i<fluid->getParticleCount(); ++i)
		{
			fluid->getHostPositions()[3 * i + 0] =
				simulationBox[0] * positionDist(*rng);
			fluid->getHostPositions()[3 * i + 1] =
				simulationBox[1] * positionDist(*rng);
			fluid->getHostPositions()[3 * i + 2] =
				simulationBox[2] * positionDist(*rng);
		}

		fluid->pushToDevice();


		THEN("`saveLogicalEntityCentersOfMassToDeviceMemory` works")
		{
			OpenMPCD::CUDA::DeviceMemoryManager dmm;
			dmm.setAutofree(true);

			MPCParticlePositionType* output;
			dmm.allocateMemory(&output, 3 * fluid->getParticleCount());

			MPCParticlePositionType* expected;
			dmm.allocateMemory(&expected, 3 * fluid->getParticleCount());

			for(unsigned int ppe = 1; ppe < 20; ++ppe)
			{ //ppe = particles per entity
				if(fluid->getParticleCount() % ppe != 0)
					continue;

				fluid->setParticlesPerLogicalentity(ppe);

				using
					OpenMPCD::CUDA::MPCFluid::DeviceCode::
						computeLogicalEntityCentersOfMass;

				OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(
					fluid->getNumberOfLogicalEntities())
						computeLogicalEntityCentersOfMass
							<<<gridSize, blockSize>>>(
								workUnitOffset, fluid->getDevicePositions(),
								fluid->getNumberOfLogicalEntities(),
								ppe, expected);
				OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
				cudaDeviceSynchronize();
				OPENMPCD_CUDA_THROW_ON_ERROR;

				boost::scoped_array<OpenMPCD::MPCParticlePositionType>
					expectedOnHost(
						new OpenMPCD::MPCParticlePositionType
							[3 * fluid->getParticleCount()]);
				dmm.copyElementsFromDeviceToHost(
					expected, expectedOnHost.get(),
					3 * fluid->getParticleCount());


				#ifdef OPENMPCD_DEBUG
					REQUIRE_THROWS_AS(
						fluid->saveLogicalEntityCentersOfMassToDeviceMemory(
							NULL),
						OpenMPCD::NULLPointerException);
				#endif


				fluid->saveLogicalEntityCentersOfMassToDeviceMemory(output);

				REQUIRE(
					dmm.elementMemoryEqualOnHostAndDevice(
						expectedOnHost.get(), output,
						3 * fluid->getParticleCount()));
			}
		}
	}
}




const std::pair<boost::shared_ptr<OpenMPCD::CUDA::Simulation>, boost::shared_ptr<OpenMPCDTest::CUDA::MPCFluid::BaseTest> >
	OpenMPCDTest::CUDA::MPCFluid::BaseTest::getSimulationAndFluid(
		const unsigned int particlesPerCell, const unsigned int (&simulationBox)[3])
{
	const boost::shared_ptr<OpenMPCD::CUDA::Simulation> sim (
		new OpenMPCD::CUDA::Simulation(getConfiguration(particlesPerCell, simulationBox), SimulationTest::getRNGSeed()));

	const unsigned int particleCount = particlesPerCell * simulationBox[0] * simulationBox[1] * simulationBox[2];

	const boost::shared_ptr<BaseTest> fluid (
		new BaseTest(sim.get(), particleCount, getMPCTimestep(), *SimulationTest::getRNG(sim.get()), sim->getDeviceMemoryManager()));

	return std::make_pair(sim, fluid);
}

void OpenMPCDTest::CUDA::MPCFluid::BaseTest::achieveTargetNumberOfParticlesInCell_host(
	OpenMPCD::CUDA::MPCFluid::Base* const fluid,
	const unsigned int targetNumber,
	const unsigned int collisionCellX,
	const unsigned int collisionCellY,
	const unsigned int collisionCellZ)
{
	//TODO
}

const OpenMPCD::Configuration OpenMPCDTest::CUDA::MPCFluid::BaseTest::getConfiguration(
	const unsigned int particlesPerCell, const unsigned int (&simulationBox)[3])
{
	OpenMPCD::Configuration config;
	config.set("initialization.particleDensity", particlesPerCell);
	config.set("initialization.particleVelocityDistribution.mean", 0.0);
	config.set("initialization.particleVelocityDistribution.standardDeviation", 1.0);

	config.set("mpc.simulationBoxSize.x", simulationBox[0]);
	config.set("mpc.simulationBoxSize.y", simulationBox[1]);
	config.set("mpc.simulationBoxSize.z", simulationBox[2]);
	config.set("mpc.timestep",            getMPCTimestep());
	config.set("mpc.srdCollisionAngle",   2.27);
	config.set("mpc.gridShiftScale",      1.0);

	config.set("bulkThermostat.targetkT", 1.0);

	config.set("mpc.sweepSize", 1);

	config.set("bulkThermostat.type", "MBS");

	config.set("boundaryConditions.LeesEdwards.shearRate", 0.0);

	config.createGroup("mpc.fluid.simple");

	return config;
}

void OpenMPCDTest::CUDA::MPCFluid::BaseTest::executeWithVariousDensitiesAndGeometries(
	void (*func)(const unsigned int, const unsigned int (&)[3]),
	const unsigned int maxParticleCount)
{
	static const unsigned int densities[] = {1, 5, 10, 30};
	static const unsigned int dimensionTriplets[][3] =
		{
				{1, 1, 1},
				{3, 3, 3},
				{10, 10, 10},
				{7, 11, 13}
		};

	int sectionCounter = 0;
	for(unsigned int densityIndex=0; densityIndex<sizeof(densities)/sizeof(densities[0]); ++densityIndex)
	{
		const unsigned int density = densities[densityIndex];

		for(unsigned int dimensionsIndex=0; dimensionsIndex<sizeof(dimensionTriplets)/sizeof(dimensionTriplets[0]); ++dimensionsIndex)
		{
			const unsigned int (&dimensions) [3] = dimensionTriplets[dimensionsIndex];

			const unsigned int particleCount =
				density * dimensions[0] * dimensions[1] * dimensions[2];

			if(maxParticleCount != 0 && maxParticleCount < particleCount)
				continue;

			SECTION("", "", sectionCounter++)
			{
				INFO("density " << density << ", dimensions " << dimensions[0] << "x" << dimensions[1] << "x" << dimensions[2]);
				func(density, dimensions);
			}
		}
	}
}

void OpenMPCDTest::CUDA::MPCFluid::BaseTest::setParticlesPerLogicalentity(
	const unsigned int value)
{
	if(getParticleCount() % value != 0)
		OPENMPCD_THROW(OpenMPCD::Exception, "");

	particlesPerLogicalEntity = value;
}
