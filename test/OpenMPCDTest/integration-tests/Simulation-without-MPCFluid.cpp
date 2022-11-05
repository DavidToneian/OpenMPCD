/**
 * @file
 * Tests that a simulation can run without an `MPCFluid`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Simulation.hpp>

static const OpenMPCD::Configuration getConfiguration()
{
	OpenMPCD::Configuration config;
	config.set("initialization.particleDensity", 0);
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

	return config;
}

static unsigned int getRNGSeed()
{
	return 12345;
}

SCENARIO(
	"`OpenMPCD::CUDA::Simulation` without an `MPCFluid`",
	"[CUDA]"
	)
{
	GIVEN("a configuration without an `MPCFluid`")
	{
		OpenMPCD::CUDA::Simulation simulation(getConfiguration(), getRNGSeed());

		THEN("one can perform sweeps")
		{
			REQUIRE_NOTHROW(simulation.sweep());
		}
	}
}
