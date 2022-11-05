/**
 * @file
 * This runs a simulation with high shear rate. Due to a bug, this resulted in a
 * crash, see Issue #1.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Simulation.hpp>

static const OpenMPCD::Configuration getConfiguration()
{
	OpenMPCD::Configuration config;
	config.set("initialization.particleDensity", 10);
	config.set("initialization.particleVelocityDistribution.mean", 0.0);
	config.set("initialization.particleVelocityDistribution.standardDeviation", 1.0);

	config.set("mpc.simulationBoxSize.x", 5);
	config.set("mpc.simulationBoxSize.y", 30);
	config.set("mpc.simulationBoxSize.z", 5);
	config.set("mpc.timestep",            0.1);
	config.set("mpc.srdCollisionAngle",   2.27);
	config.set("mpc.gridShiftScale",      1.0);

	config.set("mpc.sweepSize", 5);

	config.createGroup("mpc.fluid.simple");

	config.set("bulkThermostat.type", "MBS");
	config.set("bulkThermostat.targetkT", 1.0);

	config.set("boundaryConditions.LeesEdwards.shearRate", 20.0);

	return config;
}

static unsigned int getRNGSeed()
{
	return 12345;
}


SCENARIO(
	"Simulation with high shear rate; see Issue #1",
	"[CUDA]"
	)
{
	static const unsigned int sweepCount = 30;

	using namespace OpenMPCD;

	GIVEN("a configuration without an `MPCFluid`")
	{
		CUDA::Simulation simulation(getConfiguration(), getRNGSeed());

		for(unsigned int sweep = 0; sweep < sweepCount; ++sweep)
			simulation.sweep();
	}
}
