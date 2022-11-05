/**
 * @file
 * Tests `OpenMPCD::CUDA::Instrumentation`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Instrumentation.hpp>

#include <OpenMPCD/Utility/HostInformation.hpp>

#include <boost/filesystem.hpp>


SCENARIO(
	"`OpenMPCD::CUDA::Instrumentation::save` saves metadata",
	"[CUDA]")
{
	using namespace OpenMPCD;

	Configuration config;
	config.set("initialization.particleDensity", 10);
	config.set("initialization.particleVelocityDistribution.mean", 0.0);
	config.set("initialization.particleVelocityDistribution.standardDeviation", 1.0);

	config.set("mpc.simulationBoxSize.x", 2);
	config.set("mpc.simulationBoxSize.y", 5);
	config.set("mpc.simulationBoxSize.z", 1);
	config.set("mpc.timestep",            0.1);
	config.set("mpc.srdCollisionAngle",   2.27);
	config.set("mpc.gridShiftScale",      1.0);

	config.set("bulkThermostat.targetkT", 1.0);

	config.set("mpc.sweepSize", 3);

	config.createGroup("mpc.fluid.simple");

	config.set("bulkThermostat.type", "MBS");

	config.set("boundaryConditions.LeesEdwards.shearRate", 0.0);

	static unsigned int rngSeed = 12345;

	static unsigned int targetSweepCount = 7;

	SECTION("configure without warmup")
	{
	}

	SECTION("configure with warmup")
	{
		config.set("mpc.warmupSteps", 9);
	}

	CUDA::Simulation simulation(config, rngSeed);
	CUDA::Instrumentation instrumentation(&simulation, rngSeed, "");

	for(unsigned int sweep = 0; sweep < targetSweepCount; ++sweep)
		simulation.sweep();

	const boost::filesystem::path tempdir =
		boost::filesystem::temp_directory_path() /
		boost::filesystem::unique_path();
	boost::filesystem::create_directories(tempdir);

	instrumentation.save(tempdir.c_str());

	const std::string metadataPath = (tempdir / "metadata.txt").c_str();

	Configuration metadata(metadataPath);
	REQUIRE(metadata.read<unsigned int>("numberOfCompletedSweeps") == 7);
	REQUIRE(
		metadata.read<std::string>("executingHost")
		==
		Utility::HostInformation::getHostname());
	REQUIRE(metadata.has("startTimeUTC"));
	REQUIRE(metadata.has("endTimeUTC"));
	REQUIRE(metadata.has("DefaultCUDADevicePCIAddress"));

	boost::filesystem::remove_all(tempdir);
}
