#include <OpenMPCD/CUDA/MPCFluid/Simple.hpp>

#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/CUDA/BoundaryCondition/LeesEdwards.hpp>
#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Simple.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/Simple.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>
#include <OpenMPCD/Exceptions.hpp>

using namespace OpenMPCD::CUDA::MPCFluid;

Simple::Simple(const CUDA::Simulation* const sim, const unsigned int count,
               const FP streamingTimestep_, RNG& rng_, DeviceMemoryManager* const devMemMgr)
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


	readConfiguration();

	initializeOnHost();

	pushToDevice();

	instrumentation = new Instrumentation::Simple(sim, devMemMgr, this);
}

void Simple::readConfiguration()
{
}

void Simple::stream()
{
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(getParticleCount())
		DeviceCode::streamSimpleMPCParticle <<<blockSize, gridSize>>> (
			workUnitOffset,
			getDevicePositions(),
			getDeviceVelocities());
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
}

void Simple::initializeOnHost()
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
