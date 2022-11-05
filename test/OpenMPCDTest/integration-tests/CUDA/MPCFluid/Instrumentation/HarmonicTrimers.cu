/**
 * @file
 * Integration tests for
 * `OpenMPCD::CUDA::MPCFluid::Instrumentation::HarmonicTrimers`.
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
	config.set("initialization.particleDensity", 9);
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

	config.set("mpc.fluid.harmonicTrimers.springConstant1", 1.0);
	config.set("mpc.fluid.harmonicTrimers.springConstant2", 1.0);
	config.set("mpc.fluid.harmonicTrimers.analyticalStreaming", true);

	config.set("bulkThermostat.type", "MBS");

	config.set("boundaryConditions.LeesEdwards.shearRate", 0.0);

	const std::string groupName =
		"instrumentation.harmonicTrimers";

	config.set(groupName + ".bond1LengthHistogram.binCount", 1);
	config.set(groupName + ".bond1LengthHistogram.low", 0.0);
	config.set(groupName + ".bond1LengthHistogram.high", 1.0);

	config.set(groupName + ".bond2LengthHistogram.binCount", 1);
	config.set(groupName + ".bond2LengthHistogram.low", 0.0);
	config.set(groupName + ".bond2LengthHistogram.high", 1.0);

	config.set(groupName + ".bond1LengthSquaredHistogram.binCount", 1);
	config.set(groupName + ".bond1LengthSquaredHistogram.low", 0.0);
	config.set(groupName + ".bond1LengthSquaredHistogram.high", 1.0);

	config.set(groupName + ".bond2LengthSquaredHistogram.binCount", 1);
	config.set(groupName + ".bond2LengthSquaredHistogram.low", 0.0);
	config.set(groupName + ".bond2LengthSquaredHistogram.high", 1.0);

	config.set(groupName + ".bond1XXHistogram.binCount", 1);
	config.set(groupName + ".bond1XXHistogram.low", 0.0);
	config.set(groupName + ".bond1XXHistogram.high", 1.0);

	config.set(groupName + ".bond2XXHistogram.binCount", 1);
	config.set(groupName + ".bond2XXHistogram.low", 0.0);
	config.set(groupName + ".bond2XXHistogram.high", 1.0);

	config.set(groupName + ".bond1YYHistogram.binCount", 1);
	config.set(groupName + ".bond1YYHistogram.low", 0.0);
	config.set(groupName + ".bond1YYHistogram.high", 1.0);

	config.set(groupName + ".bond2YYHistogram.binCount", 1);
	config.set(groupName + ".bond2YYHistogram.low", 0.0);
	config.set(groupName + ".bond2YYHistogram.high", 1.0);

	config.set(groupName + ".bond1ZZHistogram.binCount", 1);
	config.set(groupName + ".bond1ZZHistogram.low", 0.0);
	config.set(groupName + ".bond1ZZHistogram.high", 1.0);

	config.set(groupName + ".bond2ZZHistogram.binCount", 1);
	config.set(groupName + ".bond2ZZHistogram.low", 0.0);
	config.set(groupName + ".bond2ZZHistogram.high", 1.0);

	config.set(groupName + ".bond1XYHistogram.binCount", 1);
	config.set(groupName + ".bond1XYHistogram.low", 0.0);
	config.set(groupName + ".bond1XYHistogram.high", 1.0);

	config.set(groupName + ".bond2XYHistogram.binCount", 1);
	config.set(groupName + ".bond2XYHistogram.low", 0.0);
	config.set(groupName + ".bond2XYHistogram.high", 1.0);

	config.set(groupName + ".bond1XXHistogram.binCount", 1);
	config.set(groupName + ".bond1XXHistogram.low", 0.0);
	config.set(groupName + ".bond1XXHistogram.high", 1.0);

	config.set(
		groupName + ".bond1XYAngleWithFlowDirectionHistogram.binCount", 1);
	config.set(groupName + ".bond1XYAngleWithFlowDirectionHistogram.low", 0.0);
	config.set(groupName + ".bond1XYAngleWithFlowDirectionHistogram.high", 1.0);

	config.set(
		groupName + ".bond2XYAngleWithFlowDirectionHistogram.binCount", 1);
	config.set(groupName + ".bond2XYAngleWithFlowDirectionHistogram.low", 0.0);
	config.set(groupName + ".bond2XYAngleWithFlowDirectionHistogram.high", 1.0);


	config.set("instrumentation.selfDiffusionCoefficient.measurementTime", 1.0);

	config.createList("instrumentation.fourierTransformedVelocity.k_n");

	config.set("instrumentation.velocityAutocorrelation.measurementTime", 1.0);
	config.set("instrumentation.velocityAutocorrelation.snapshotCount", 1);


	return config;
}

static unsigned int getRNGSeed()
{
	return 12345;
}



SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::Instrumentation::"
		"HarmonicTrimers` can be configured in "
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
