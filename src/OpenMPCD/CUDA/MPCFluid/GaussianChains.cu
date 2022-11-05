#include <OpenMPCD/CUDA/MPCFluid/GaussianChains.hpp>

#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/CUDA/BoundaryCondition/LeesEdwards.hpp>
#include <OpenMPCD/CUDA/DeviceCode/LeesEdwardsBoundaryConditions.hpp>
#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/GaussianChains.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/GaussianChains.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>
#include <OpenMPCD/Exceptions.hpp>

using namespace OpenMPCD;
using namespace OpenMPCD::CUDA;
using namespace OpenMPCD::CUDA::MPCFluid;

GaussianChains::GaussianChains(const CUDA::Simulation* const sim, const unsigned int count,
                               const FP streamingTimestep_, RNG& rng_,
                               DeviceMemoryManager* const devMemMgr)
	: Base(sim, count, streamingTimestep_, rng_, devMemMgr),
	  d_velocityVerletAccelerationBuffer(NULL)
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

	if(particlesPerChain == 0)
		OPENMPCD_THROW(InvalidConfigurationException, "Cannot have Gaussian Chains of length 0.");

	if(getParticleCount() % particlesPerChain != 0)
		OPENMPCD_THROW(InvalidConfigurationException, "The total number of MPC particles is not a "
		                                             "multiple of the number of MPC particles per "
		                                             "chain in MPCFluid::GaussianChains.");

	initializeOnHost();

	pushToDevice();

	instrumentation = new Instrumentation::GaussianChains(particlesPerChain, sim, devMemMgr, this);

	deviceMemoryManager->allocateMemory(&d_velocityVerletAccelerationBuffer,  3 * getParticleCount());
}

GaussianChains::~GaussianChains()
{
	deviceMemoryManager->freeMemory(d_velocityVerletAccelerationBuffer);
}

void GaussianChains::readConfiguration()
{
	const Configuration& config = simulation->getConfiguration();

	config.read("mpc.fluid.gaussianChains.particlesPerChain", &particlesPerChain);
	config.read("mpc.fluid.gaussianChains.springConstant", &springConstant);
	config.read("mpc.fluid.gaussianChains.mdStepCount", &mdStepCount);
}

void GaussianChains::stream()
{
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(getParticleCount() / particlesPerChain)
		DeviceCode::streamGaussianChainsVelocityVerlet<<<gridSize, blockSize>>>(
			workUnitOffset,
			d_mpcParticlePositions,
			d_mpcParticleVelocities,
			d_velocityVerletAccelerationBuffer,
			particlesPerChain,
			springConstant / mpcParticleMass,
			streamingTimestep / mdStepCount,
			mdStepCount);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

void GaussianChains::initializeOnHost()
{
	initializeVelocitiesOnHost();

	boost::random::uniform_01<FP> posDist;

	for(unsigned int pID = 0; pID < getParticleCount(); pID += particlesPerChain)
	{
		mpcParticlePositions[pID * 3 + 0] = posDist(rng) * simulation->getSimulationBoxSizeX();
		mpcParticlePositions[pID * 3 + 1] = posDist(rng) * simulation->getSimulationBoxSizeY();
		mpcParticlePositions[pID * 3 + 2] = posDist(rng) * simulation->getSimulationBoxSizeZ();

		const RemotelyStoredVector<MPCParticlePositionType> lastPosition(mpcParticlePositions, pID);

		for(unsigned int partnerOffset = 1; partnerOffset < particlesPerChain; ++partnerOffset)
		{
			const Vector3D<MPCParticlePositionType> displacement
				= Vector3D<MPCParticlePositionType>::getRandomUnitVector(rng);

			const Vector3D<MPCParticlePositionType> partnerPosition = lastPosition + displacement;

			if(	partnerPosition.getX() >= simulation->getSimulationBoxSizeX() ||
				partnerPosition.getY() >= simulation->getSimulationBoxSizeY() ||
				partnerPosition.getZ() >= simulation->getSimulationBoxSizeZ() ||
				partnerPosition.getX() < 0 || partnerPosition.getY() < 0 || partnerPosition.getZ() < 0)
			{
				//The new particle would start outside of the box,
				//thus gaining in velocity under Lees-Edwards boundary conditions.
				//Choose new positions instead.
				--partnerOffset;
				continue;
			}


			mpcParticlePositions[(pID + partnerOffset) * 3 + 0] = partnerPosition.getX();
			mpcParticlePositions[(pID + partnerOffset) * 3 + 1] = partnerPosition.getY();
			mpcParticlePositions[(pID + partnerOffset) * 3 + 2] = partnerPosition.getZ();
		}
	}
}
