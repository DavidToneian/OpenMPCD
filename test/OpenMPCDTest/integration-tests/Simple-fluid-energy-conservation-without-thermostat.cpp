/**
 * @file
 * This test runs a simulation with a simple fluid, and checks whether energy is
 * conserved.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Simulation.hpp>

#include <fstream>
#include <limits>

using namespace OpenMPCD;

static const Configuration getConfiguration()
{
	OpenMPCD::Configuration config;
	config.set("initialization.particleDensity", 10);
	config.set("initialization.particleVelocityDistribution.mean", 0.0);
	config.set("initialization.particleVelocityDistribution.standardDeviation", 1.0);

	config.set("mpc.simulationBoxSize.x", 2);
	config.set("mpc.simulationBoxSize.y", 5);
	config.set("mpc.simulationBoxSize.z", 1);
	config.set("mpc.timestep",            0.1);
	config.set("mpc.srdCollisionAngle",   2.27);
	config.set("mpc.gridShiftScale",      1.0);

	config.set("mpc.sweepSize", 10);

	config.createGroup("mpc.fluid.simple");

	config.set("boundaryConditions.LeesEdwards.shearRate", 0.0);

	return config;
}

static unsigned int getRNGSeed()
{
	return 12345;
}


static const FP getEnergy(const CUDA::MPCFluid::Base& fluid)
{
	fluid.fetchFromDevice();

	FP ret = 0;

	for(unsigned int pID = 0; pID < fluid.getParticleCount(); ++pID)
		ret += fluid.getVelocity(pID).getMagnitudeSquared();

	return ret * fluid.getParticleMass() / 2.0;
}

SCENARIO(
	"CUDA, Simple MPC Fluid, check for conservation of energy "
		"(without thermostat)",
	"[CUDA]"
	)
{
	static unsigned int sweepCount = 100;

	CUDA::Simulation simulation(getConfiguration(), getRNGSeed());

	const FP initialEnergy = getEnergy(simulation.getMPCFluid());

	for(unsigned int iteration = 0; iteration < sweepCount; ++iteration)
	{
		simulation.sweep();

		REQUIRE(initialEnergy == Approx(getEnergy(simulation.getMPCFluid())));
	}
}
