/**
 * @file
 * This test runs a simulation with a simple fluid, and checks whether the
 * density profile output file is what is expected.
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

	config.createGroup("mpc.fluid.simple");

	config.set("bulkThermostat.type", "MBS");

	config.set("boundaryConditions.LeesEdwards.shearRate", 0.0);

	config.createGroup("instrumentation.densityProfile");

	return config;
}

static unsigned int getRNGSeed()
{
	return 12345;
}



static void requireFilesAreApproximatelyEqual(
	const char* const newPath, const char* const referencePath)
{
	typedef OpenMPCD::FP T;

	REQUIRE(boost::filesystem::is_regular_file(newPath));
	REQUIRE(boost::filesystem::is_regular_file(referencePath));

	std::ifstream file(newPath);
	std::ifstream referenceFile(referencePath);


	T posX, posY, posZ;
	T density;

	T ref_posX, ref_posY, ref_posZ;
	T ref_density;

	while(
		file
		>> posX >> posY >> posZ
		>> density)
	{
		REQUIRE(referenceFile.good());

		referenceFile
			>> ref_posX >> ref_posY >> ref_posZ
			>> ref_density;


		REQUIRE(posX == ref_posX);
		REQUIRE(posY == ref_posY);
		REQUIRE(posZ == ref_posZ);

		REQUIRE(density == ref_density);
	}

	std::string tmp;
	file >> tmp;
	REQUIRE(file.eof());

	referenceFile >> tmp;
	REQUIRE(referenceFile.eof());
}


SCENARIO(
	"DensityProfile integration test, no cell subdivision",
	"[CUDA]"
	)
{
	static const char* const referenceFilePath =
		"test/data/integration-tests/DensityProfile/densityProfile.data";
	static unsigned int sweepCount = 100;

	const boost::filesystem::path tempdir =
		boost::filesystem::temp_directory_path() /
		boost::filesystem::unique_path();
	boost::filesystem::create_directories(tempdir);

	{
		const unsigned int rngSeed = getRNGSeed();
		static const char* const gitCommitIdentifier = "";

		CUDA::Simulation simulation(getConfiguration(), rngSeed);
		CUDA::Instrumentation instrumentation(
			&simulation, rngSeed, gitCommitIdentifier);
		instrumentation.setAutosave(tempdir.string());

		for(unsigned int i = 0; i < sweepCount; ++i)
		{
			simulation.sweep();
			instrumentation.measure();
		}
	}


	requireFilesAreApproximatelyEqual(
		(tempdir / "densityProfile.data").c_str(),
		referenceFilePath);


	boost::filesystem::remove_all(tempdir);
}


SCENARIO(
	"DensityProfile integration test, with cell subdivision",
	"[CUDA]"
	)
{
	static const char* const referenceFilePath =
		"test/data/integration-tests/DensityProfile/"
		"densityProfile-cellSubdivision.data";
	static unsigned int sweepCount = 100;

	const boost::filesystem::path tempdir =
		boost::filesystem::temp_directory_path() /
		boost::filesystem::unique_path();
	boost::filesystem::create_directories(tempdir);

	{
		const unsigned int rngSeed = getRNGSeed();
		static const char* const gitCommitIdentifier = "";

		Configuration config = getConfiguration();
		config.set("instrumentation.densityProfile.cellSubdivision.x", 1);
		config.set("instrumentation.densityProfile.cellSubdivision.y", 2);
		config.set("instrumentation.densityProfile.cellSubdivision.z", 3);

		CUDA::Simulation simulation(config, rngSeed);
		CUDA::Instrumentation instrumentation(
			&simulation, rngSeed, gitCommitIdentifier);
		instrumentation.setAutosave(tempdir.string());

		for(unsigned int i = 0; i < sweepCount; ++i)
		{
			simulation.sweep();
			instrumentation.measure();
		}
	}


	requireFilesAreApproximatelyEqual(
		(tempdir / "densityProfile.data").c_str(),
		referenceFilePath);


	boost::filesystem::remove_all(tempdir);
}
