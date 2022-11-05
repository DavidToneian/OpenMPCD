/**
 * @file
 * This test runs a simulation with a simple fluid, but without a thermostat and
 * without collisions. The result should be ballistic motion of the particles.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Simulation.hpp>

#include <vector>

using namespace OpenMPCD;

static const FP timestep = 0.1;

static const Configuration getConfiguration()
{
	OpenMPCD::Configuration config;
	config.set("initialization.particleDensity", 10);
	config.set("initialization.particleVelocityDistribution.mean", 0.0);
	config.set("initialization.particleVelocityDistribution.standardDeviation", 1.0);

	config.set("mpc.simulationBoxSize.x", 2);
	config.set("mpc.simulationBoxSize.y", 5);
	config.set("mpc.simulationBoxSize.z", 1);
	config.set("mpc.timestep",            timestep);
	config.set("mpc.srdCollisionAngle",   0.0);
	config.set("mpc.gridShiftScale",      1.0);

	config.set("mpc.sweepSize", 1);

	config.createGroup("mpc.fluid.simple");

	config.set("boundaryConditions.LeesEdwards.shearRate", 0.0);

	return config;
}

static unsigned int getRNGSeed()
{
	return 12345;
}


SCENARIO(
	"CUDA, Simple MPC Fluid, no thermostat, no collision",
	"[CUDA]"
	)
{
	static unsigned int sweepCount = 3;

	CUDA::Simulation simulation(getConfiguration(), getRNGSeed());

	const unsigned int particleCount =
		simulation.getMPCFluid().getParticleCount();

	simulation.getMPCFluid().fetchFromDevice();

	std::vector<Vector3D<MPCParticlePositionType> > initialPositions;
	initialPositions.reserve(particleCount);
	for(unsigned int p = 0; p < particleCount; ++p)
		initialPositions.push_back(simulation.getMPCFluid().getPosition(p));

	std::vector<Vector3D<MPCParticleVelocityType> > initialVelocities;
	initialVelocities.reserve(particleCount);
	for(unsigned int p = 0; p < particleCount; ++p)
		initialVelocities.push_back(simulation.getMPCFluid().getVelocity(p));

	for(unsigned int iteration = 0; iteration < sweepCount; ++iteration)
	{
		simulation.sweep();

		simulation.getMPCFluid().fetchFromDevice();

		for(unsigned int p = 0; p < particleCount; ++p)
		{
			REQUIRE(
				initialVelocities[p] ==
				simulation.getMPCFluid().getVelocity(p));

			const Vector3D<MPCParticlePositionType> expected =
				initialPositions[p] +
				(iteration + 1) * timestep * initialVelocities[p];

			const Vector3D<MPCParticlePositionType> actual =
				simulation.getMPCFluid().getPosition(p);

			REQUIRE(expected.getX() == Approx(actual.getX()));
			REQUIRE(expected.getY() == Approx(actual.getY()));
			REQUIRE(expected.getZ() == Approx(actual.getZ()));
		}

		std::vector<Vector3D<MPCParticleVelocityType> > initialVelocities;
		initialVelocities.reserve(particleCount);
		for(unsigned int p = 0; p < particleCount; ++p)
			initialVelocities.push_back(simulation.getMPCFluid().getVelocity(p));
	}
}
