/**
 * @file
 * This test runs a simulation with a `GaussianDumbbells`, and checks whether
 * the results correspond to pre-recorded results.
 * The result, of course, depends on implementation details (e.g. the order in
 * which random variables are generated) that are a priori irrelevant to the
 * correctness of the implementation. However, this test is useful to detect
 * unintended deviations of known behavior. If, however, the implementation is
 * changed in a way that makes this test fail, that does not necessarily mean
 * that the new implementation is invalid.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Simulation.hpp>

#include <limits>

static const OpenMPCD::Configuration getConfiguration()
{
	OpenMPCD::Configuration config;
	config.set("initialization.particleDensity", 10);
	config.set("initialization.particleVelocityDistribution.mean", 0.0);
	config.set("initialization.particleVelocityDistribution.standardDeviation", 1.0);

	config.set("mpc.simulationBoxSize.x", 5);
	config.set("mpc.simulationBoxSize.y", 5);
	config.set("mpc.simulationBoxSize.z", 5);
	config.set("mpc.timestep",            0.1);
	config.set("mpc.srdCollisionAngle",   2.27);
	config.set("mpc.gridShiftScale",      1.0);

	config.set("bulkThermostat.targetkT", 1.5);

	config.set("mpc.sweepSize", 1);

	config.set("initialization.dumbbell.relativePosition", "perfect");
	config.set("mpc.fluid.dumbbell.rootMeanSquareLength", 3.0);
	config.set("mpc.fluid.dumbbell.zeroShearRelaxationTime", 26.0);
	config.set("mpc.fluid.dumbbell.analyticalStreaming", false);
	config.set("mpc.fluid.dumbbell.mdStepCount", 3);

	config.set("instrumentation.dumbbellBondLengthHistogram.binCount", 10);
	config.set("instrumentation.dumbbellBondLengthHistogram.low", 0.0);
	config.set("instrumentation.dumbbellBondLengthHistogram.high", 10.0);

	config.set("instrumentation.dumbbellBondLengthSquaredHistogram.binCount", 10);
	config.set("instrumentation.dumbbellBondLengthSquaredHistogram.low", 0.0);
	config.set("instrumentation.dumbbellBondLengthSquaredHistogram.high", 10.0);

	config.set("instrumentation.dumbbellBondXHistogram.binCount", 10);
	config.set("instrumentation.dumbbellBondXHistogram.low", 0.0);
	config.set("instrumentation.dumbbellBondXHistogram.high", 10.0);

	config.set("instrumentation.dumbbellBondYHistogram.binCount", 10);
	config.set("instrumentation.dumbbellBondYHistogram.low", 0.0);
	config.set("instrumentation.dumbbellBondYHistogram.high", 10.0);

	config.set("instrumentation.dumbbellBondZHistogram.binCount", 10);
	config.set("instrumentation.dumbbellBondZHistogram.low", 0.0);
	config.set("instrumentation.dumbbellBondZHistogram.high", 10.0);

	config.set("instrumentation.dumbbellBondXXHistogram.binCount", 10);
	config.set("instrumentation.dumbbellBondXXHistogram.low", 0.0);
	config.set("instrumentation.dumbbellBondXXHistogram.high", 10.0);

	config.set("instrumentation.dumbbellBondYYHistogram.binCount", 10);
	config.set("instrumentation.dumbbellBondYYHistogram.low", 0.0);
	config.set("instrumentation.dumbbellBondYYHistogram.high", 10.0);

	config.set("instrumentation.dumbbellBondZZHistogram.binCount", 10);
	config.set("instrumentation.dumbbellBondZZHistogram.low", 0.0);
	config.set("instrumentation.dumbbellBondZZHistogram.high", 10.0);

	config.set("instrumentation.dumbbellBondXYHistogram.binCount", 10);
	config.set("instrumentation.dumbbellBondXYHistogram.low", 0.0);
	config.set("instrumentation.dumbbellBondXYHistogram.high", 10.0);

	config.set("instrumentation.dumbbellBondAngleWithFlowDirectionHistogram.binCount", 10);
	config.set("instrumentation.dumbbellBondAngleWithFlowDirectionHistogram.low", 0.0);
	config.set("instrumentation.dumbbellBondAngleWithFlowDirectionHistogram.high", 10.0);

	config.set("instrumentation.dumbbellBondXYAngleWithFlowDirectionHistogram.binCount", 10);
	config.set("instrumentation.dumbbellBondXYAngleWithFlowDirectionHistogram.low", 0.0);
	config.set("instrumentation.dumbbellBondXYAngleWithFlowDirectionHistogram.high", 10.0);

	config.createList("instrumentation.fourierTransformedVelocity.k_n");
	config.set("instrumentation.velocityAutocorrelation.measurementTime", 1.0);
	config.set("instrumentation.velocityAutocorrelation.snapshotCount", 1);


	config.set("bulkThermostat.type", "MBS");

	config.set("boundaryConditions.LeesEdwards.shearRate", 0.1);

	return config;
}

static unsigned int getRNGSeed()
{
	return 12345;
}

template<typename T>
static Catch::Detail::Approx Approx(const T value)
{
	return
		Catch::Detail::Approx(value).
		epsilon(std::numeric_limits<T>::epsilon() * 7);
}

SCENARIO(
	"CUDA, GaussianDumbbells, Velocity-Verlet Streaming, "
		"Reference Simulation 1",
	"[CUDA]"
	)
{
	static const char* const path =
		"test/data/reference-simulations/"
		"GaussianDumbbells-VelocityVerletStreaming-1.vtf";

	static const unsigned int iterationCount = 3;

	using namespace OpenMPCD;

	GIVEN("a configuration without an `MPCFluid`")
	{
		CUDA::Simulation simulation(getConfiguration(), getRNGSeed());
		const CUDA::MPCFluid::Base& fluid = simulation.getMPCFluid();

		VTFSnapshotFile snapshot(path);

		/* to create reference data:
		snapshot.declareAtoms(simulation.getMPCFluid().getParticleCount());

		for(unsigned int iteration = 0; iteration < iterationCount; ++iteration)
		{
			if(iteration != 0)
				simulation.sweep();

			fluid.writeToSnapshot(&snapshot);
		}
		return;
		*/

		const std::size_t particleCount = fluid.getParticleCount();
		const std::size_t coordinateCount = 3 * particleCount;

		REQUIRE(snapshot.getNumberOfAtoms() == particleCount);

		boost::scoped_array<MPCParticlePositionType>
			positions(new MPCParticlePositionType[coordinateCount]);
		boost::scoped_array<OpenMPCD::MPCParticleVelocityType>
			velocities(new MPCParticleVelocityType[coordinateCount]);

		for(unsigned int iteration = 0; iteration < iterationCount; ++iteration)
		{
			if(iteration != 0)
				simulation.sweep();

			bool velocitiesEncountered = false;
			REQUIRE(
				snapshot.readTimestepBlock(
					positions.get(), velocities.get(), &velocitiesEncountered));
			REQUIRE(velocitiesEncountered);

			fluid.fetchFromDevice();
			for(std::size_t particle = 0; particle < particleCount; ++particle)
			{
				const RemotelyStoredVector<const MPCParticlePositionType>
					r = fluid.getPosition(particle);
				const RemotelyStoredVector<const MPCParticleVelocityType>
					v = fluid.getVelocity(particle);

				REQUIRE(r.getX() == Approx(positions[3 * particle + 0]));
				REQUIRE(r.getY() == Approx(positions[3 * particle + 1]));
				REQUIRE(r.getZ() == Approx(positions[3 * particle + 2]));

				REQUIRE(v.getX() == Approx(velocities[3 * particle + 0]));
				REQUIRE(v.getY() == Approx(velocities[3 * particle + 1]));
				REQUIRE(v.getZ() == Approx(velocities[3 * particle + 2]));
			}
		}
	}
}
