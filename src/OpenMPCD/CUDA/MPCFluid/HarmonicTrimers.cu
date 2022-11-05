#include <OpenMPCD/CUDA/MPCFluid/HarmonicTrimers.hpp>

#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/CUDA/BoundaryCondition/LeesEdwards.hpp>
#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/HarmonicTrimers.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/HarmonicTrimers.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>
#include <OpenMPCD/Exceptions.hpp>

using namespace OpenMPCD::CUDA::MPCFluid;

HarmonicTrimers::HarmonicTrimers(const CUDA::Simulation* const sim, const unsigned int count,
                                 const FP streamingTimestep_, RNG& rng_,
                                 DeviceMemoryManager* const devMemMgr)
	: Base(sim, count, streamingTimestep_, rng_, devMemMgr)
{
	if(
		dynamic_cast<const BoundaryCondition::LeesEdwards*>(
			simulation->getBoundaryConditions())
		== NULL )
	{
		OPENMPCD_THROW(
			UnimplementedException,
			"Currently, only Lees-Edwards boundary conditions are "
			"supported.");
	}


	if(getParticleCount() % 3 != 0)
		OPENMPCD_THROW(InvalidConfigurationException, "For trimer simulations, the total number of "
		                                             "particles has to be a multiple of 3.");

	readConfiguration();

	initializeOnHost();

	pushToDevice();

	instrumentation = new Instrumentation::HarmonicTrimers(sim, devMemMgr, this);
}

void HarmonicTrimers::readConfiguration()
{
	const Configuration& config = simulation->getConfiguration();

	config.read("mpc.fluid.harmonicTrimers.springConstant1", &springConstant1);
	config.read("mpc.fluid.harmonicTrimers.springConstant2", &springConstant2);

	config.read("mpc.fluid.harmonicTrimers.analyticalStreaming", &streamAnalyticallyFlag);

	if(!streamAnalyticallyFlag)
		config.read("mpc.fluid.harmonicTrimers.mdStepCount", &mdStepCount);
}

void HarmonicTrimers::stream()
{
	if(streamAnalyticallyFlag)
	{
		OPENMPCD_THROW(UnimplementedException, "Analytic streaming of harmonic trimers is not supported.");
	}
	else
	{
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(getParticleCount()/3)
			DeviceCode::streamHarmonicTrimerVelocityVerlet <<<blockSize, gridSize>>> (
				workUnitOffset,
				getDevicePositions(),
				getDeviceVelocities(),
				springConstant1 / mpcParticleMass,
				springConstant2 / mpcParticleMass,
				streamingTimestep / mdStepCount,
				mdStepCount);
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	}

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

void HarmonicTrimers::initializeOnHost()
{
	initializeVelocitiesOnHost();

	boost::random::uniform_01<FP> posDist;

	for(unsigned int pID = 0; pID < getParticleCount(); ++pID)
	{
		mpcParticlePositions[pID * 3 + 0] = posDist(rng) * simulation->getSimulationBoxSizeX();
		mpcParticlePositions[pID * 3 + 1] = posDist(rng) * simulation->getSimulationBoxSizeY();
		mpcParticlePositions[pID * 3 + 2] = posDist(rng) * simulation->getSimulationBoxSizeZ();
	}
}
