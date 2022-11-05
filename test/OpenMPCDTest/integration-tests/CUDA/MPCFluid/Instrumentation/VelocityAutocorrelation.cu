/**
 * @file
 * Integration tests for
 * `OpenMPCD::CUDA::MPCFluid::Instrumentation::VelocityAutocorrelation`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Simulation.hpp>
#include <OpenMPCD/CUDA/Instrumentation.hpp>


#include <boost/filesystem.hpp>

#include <fstream>
#include <limits>

using namespace OpenMPCD;


static const Configuration getConfiguration()
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

	config.set("bulkThermostat.targetkT", 3.1415);

	config.set("mpc.sweepSize", 10);

	config.set("mpc.fluid.gaussianChains.particlesPerChain", 5);
	config.set("mpc.fluid.gaussianChains.springConstant", 1.5);
	config.set("mpc.fluid.gaussianChains.mdStepCount", 3);

	config.set("bulkThermostat.type", "MBS");

	config.set("boundaryConditions.LeesEdwards.shearRate", 0.0);

	const std::string groupName =
		"instrumentation.velocityAutocorrelation";
	config.set(groupName + ".snapshotCount", 1);
	config.set(groupName + ".measurementTime", 10.0);

	return config;
}

static unsigned int getRNGSeed()
{
	return 12345;
}



SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"VelocityAutocorrelation` can be configured in "
		"`Simulation` instance",
	"[CUDA]"
	)
{
	{
		const unsigned int rngSeed = getRNGSeed();
		static const char* const gitCommitIdentifier = "";

		CUDA::Simulation simulation(getConfiguration(), rngSeed);
		CUDA::Instrumentation instrumentation(
			&simulation, rngSeed, gitCommitIdentifier);
	}
}
