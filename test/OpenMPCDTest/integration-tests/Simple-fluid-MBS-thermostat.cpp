/**
 * @file
 * This test runs a simulation with a simple fluid, and checks whether the
 * Maxwell-Boltzmann-Scaling (MBS) thermostat works.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Simulation.hpp>

#include <fstream>
#include <limits>

using namespace OpenMPCD;

static const FP relativeErrorTolerance = 0.01;

static const FP targetkT = 3.1415;

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

	config.set("mpc.sweepSize", 10);

	config.createGroup("mpc.fluid.simple");

	config.set("bulkThermostat.type", "MBS");

	config.set("boundaryConditions.LeesEdwards.shearRate", 0.0);

	return config;
}

static unsigned int getRNGSeed()
{
	return 12345;
}


static const Vector3D<MPCParticleVelocityType>
getTotalMomentum(const CUDA::MPCFluid::Base& fluid)
{
	Vector3D<MPCParticleVelocityType> ret(0, 0, 0);

	for(unsigned int pID = 0; pID < fluid.getParticleCount(); ++pID)
		ret += fluid.getVelocity(pID) * fluid.getParticleMass();

	return ret;
}

static FP
getTotalEnergy(const CUDA::MPCFluid::Base& fluid)
{
	FP ret = 0;

	for(unsigned int pID = 0; pID < fluid.getParticleCount(); ++pID)
	{
		ret +=
			fluid.getVelocity(pID).getMagnitudeSquared() *
			fluid.getParticleMass() / 2;
	}

	return ret;
}

SCENARIO(
	"CUDA, Simple MPC Fluid, MBS thermostat",
	"[CUDA]"
	)
{
	static unsigned int warmupSweepCount = 1000;
	static unsigned int sweepCount = 1000;

	CUDA::Simulation simulation(getConfiguration(), getRNGSeed());

	simulation.getMPCFluid().fetchFromDevice();

	const Vector3D<MPCParticleVelocityType> initialMomentum =
		getTotalMomentum(simulation.getMPCFluid());

	REQUIRE(initialMomentum.getX() == Approx(0));
	REQUIRE(initialMomentum.getY() == Approx(0));
	REQUIRE(initialMomentum.getZ() == Approx(0));

	for(unsigned int iteration = 0; iteration < warmupSweepCount; ++iteration)
		simulation.sweep();

	FP accumulatedEnergyPerParticle = 0;
	for(unsigned int iteration = 0; iteration < sweepCount; ++iteration)
	{
		simulation.sweep();

		simulation.getMPCFluid().fetchFromDevice();

		const FP totalEnergy = getTotalEnergy(simulation.getMPCFluid());

		accumulatedEnergyPerParticle +=
			totalEnergy / simulation.getMPCFluid().getParticleCount();
	}

	const FP energyPerParticle = accumulatedEnergyPerParticle / sweepCount;
	const FP targetEnergyPerParticle = 3.0 / 2.0 * targetkT;

	FP relativeError =
		(targetEnergyPerParticle - energyPerParticle) / targetEnergyPerParticle;
	if(relativeError < 0)
		relativeError *= -1;

	REQUIRE(relativeError < relativeErrorTolerance);
}
