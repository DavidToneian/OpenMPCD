/**
 * @file
 * Integration tests for
 * `OpenMPCD::CUDA::MPCFluid::Instrumentation::GaussianDumbbells`.
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

	config.set("mpc.fluid.dumbbell.analyticalStreaming", true);
	config.set("mpc.fluid.dumbbell.rootMeanSquareLength", 1.0);
	config.set("mpc.fluid.dumbbell.zeroShearRelaxationTime", 1.0);

	config.set("bulkThermostat.type", "MBS");

	config.set("boundaryConditions.LeesEdwards.shearRate", 0.0);

	const std::string groupName =
		"instrumentation";

	config.set(groupName + ".dumbbellBondLengthHistogram.binCount", 1);
	config.set(groupName + ".dumbbellBondLengthHistogram.low", 0.0);
	config.set(groupName + ".dumbbellBondLengthHistogram.high", 1.0);

	config.set(groupName + ".dumbbellBondLengthSquaredHistogram.binCount", 1);
	config.set(groupName + ".dumbbellBondLengthSquaredHistogram.low", 0.0);
	config.set(groupName + ".dumbbellBondLengthSquaredHistogram.high", 1.0);

	config.set(groupName + ".dumbbellBondXHistogram.binCount", 1);
	config.set(groupName + ".dumbbellBondXHistogram.low", 0.0);
	config.set(groupName + ".dumbbellBondXHistogram.high", 1.0);

	config.set(groupName + ".dumbbellBondYHistogram.binCount", 1);
	config.set(groupName + ".dumbbellBondYHistogram.low", 0.0);
	config.set(groupName + ".dumbbellBondYHistogram.high", 1.0);

	config.set(groupName + ".dumbbellBondZHistogram.binCount", 1);
	config.set(groupName + ".dumbbellBondZHistogram.low", 0.0);
	config.set(groupName + ".dumbbellBondZHistogram.high", 1.0);

	config.set(groupName + ".dumbbellBondXXHistogram.binCount", 1);
	config.set(groupName + ".dumbbellBondXXHistogram.low", 0.0);
	config.set(groupName + ".dumbbellBondXXHistogram.high", 1.0);

	config.set(groupName + ".dumbbellBondYYHistogram.binCount", 1);
	config.set(groupName + ".dumbbellBondYYHistogram.low", 0.0);
	config.set(groupName + ".dumbbellBondYYHistogram.high", 1.0);

	config.set(groupName + ".dumbbellBondZZHistogram.binCount", 1);
	config.set(groupName + ".dumbbellBondZZHistogram.low", 0.0);
	config.set(groupName + ".dumbbellBondZZHistogram.high", 1.0);

	config.set(groupName + ".dumbbellBondXYHistogram.binCount", 1);
	config.set(groupName + ".dumbbellBondXYHistogram.low", 0.0);
	config.set(groupName + ".dumbbellBondXYHistogram.high", 1.0);

	config.set(
		groupName + ".dumbbellBondAngleWithFlowDirectionHistogram.binCount",
		1);
	config.set(
		groupName + ".dumbbellBondAngleWithFlowDirectionHistogram.low", 0.0);
	config.set(
		groupName + ".dumbbellBondAngleWithFlowDirectionHistogram.high", 1.0);

	config.set(
		groupName + ".dumbbellBondXYAngleWithFlowDirectionHistogram.binCount",
		1);
	config.set(
		groupName + ".dumbbellBondXYAngleWithFlowDirectionHistogram.low", 0.0);
	config.set(
		groupName + ".dumbbellBondXYAngleWithFlowDirectionHistogram.high", 1.0);


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
		"GaussianDumbbells` can be configured in "
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
