/**
 * @file
 * This test runs a performance test with a simple fluid and no thermostat.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Simulation.hpp>

#include <OpenMPCD/Profiling/Stopwatch.hpp>

#include <fstream>

using namespace OpenMPCD;

static const FP relativeErrorTolerance = 0.01;

static const FP targetkT = 3.1415;

static const char* const outputPath =
	"test/output/performance-tests/Simple-fluid-no-thermostat.txt";

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

	config.set("bulkThermostat.targetkT", targetkT);

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
	"Performance Test: CUDA, Simple MPC Fluid, no thermostat",
	"[CUDA]"
	)
{
	static unsigned int warmupSweepCount = 100;
	static unsigned int sweepCount = 10000;

	std::ofstream file(outputPath, std::ios_base::out | std::ios_base::trunc);

	CUDA::Simulation simulation(getConfiguration(), getRNGSeed());

	for(unsigned int iteration = 0; iteration < warmupSweepCount; ++iteration)
		simulation.sweep();

	OpenMPCD::Profiling::Stopwatch stopwatch;
	for(unsigned int iteration = 0; iteration < sweepCount; ++iteration)
		simulation.sweep();

	file << stopwatch.getElapsedMicroseconds();
}
