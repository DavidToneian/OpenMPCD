/**
 * @file
 * This test runs a simulation with a simple fluid, and checks whether linear
 * momentum is conserved, and additionally that initially, it is (almost) zero.
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

	config.set("bulkThermostat.targetkT", 1.0);

	config.set("mpc.sweepSize", 10);

	config.createGroup("mpc.fluid.simple");

	config.set("bulkThermostat.type", "MBS");

	config.set("boundaryConditions.LeesEdwards.shearRate", 0.5);

	return config;
}

static unsigned int getRNGSeed()
{
	return 12345;
}


static const Vector3D<MPCParticleVelocityType>
getTotalMomentum(const CUDA::MPCFluid::Base& fluid)
{
	fluid.fetchFromDevice();

	Vector3D<MPCParticleVelocityType> ret(0, 0, 0);

	for(unsigned int pID = 0; pID < fluid.getParticleCount(); ++pID)
		ret += fluid.getVelocity(pID) * fluid.getParticleMass();

	return ret;
}

SCENARIO(
	"CUDA, Simple MPC Fluid, check for conservation of linear momentum",
	"[CUDA]"
	)
{
	static unsigned int sweepCount = 100;

	CUDA::Simulation simulation(getConfiguration(), getRNGSeed());

	const Vector3D<MPCParticleVelocityType> initialMomentum =
		getTotalMomentum(simulation.getMPCFluid());

	REQUIRE(initialMomentum.getX() == Approx(0));
	REQUIRE(initialMomentum.getY() == Approx(0));
	REQUIRE(initialMomentum.getZ() == Approx(0));

	for(unsigned int iteration = 0; iteration < sweepCount; ++iteration)
	{
		simulation.sweep();

		const Vector3D<MPCParticleVelocityType> totalMomentum =
				getTotalMomentum(simulation.getMPCFluid());

		REQUIRE(totalMomentum.getX() == Approx(initialMomentum.getX()));
		REQUIRE(totalMomentum.getY() == Approx(initialMomentum.getY()));
		REQUIRE(totalMomentum.getZ() == Approx(initialMomentum.getZ()));
	}
}
