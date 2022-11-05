#include <OpenMPCD/CUDA/MPCFluid/GaussianRods.hpp>

#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/CUDA/BoundaryCondition/LeesEdwards.hpp>
#include <OpenMPCD/CUDA/DeviceCode/LeesEdwardsBoundaryConditions.hpp>
#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/GaussianRods.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/GaussianRods.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>
#include <OpenMPCD/Exceptions.hpp>

using namespace OpenMPCD;
using namespace OpenMPCD::CUDA;
using namespace OpenMPCD::CUDA::MPCFluid;

GaussianRods::GaussianRods(const CUDA::Simulation* const sim, const unsigned int count,
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


	if(getParticleCount() % 2 != 0)
		OPENMPCD_THROW(InvalidConfigurationException, "For GaussianRods simulations, the total number of "
		                                             "particles has to be even.");

	readConfiguration();

	initializeOnHost();

	pushToDevice();

	instrumentation = new Instrumentation::GaussianRods(sim, devMemMgr, this);
}

void GaussianRods::readConfiguration()
{
	const Configuration& config = simulation->getConfiguration();

	config.read("mpc.fluid.gaussianRods.meanBondLength", &meanBondLength);
	config.read("mpc.fluid.gaussianRods.springConstant", &springConstant);
	config.read("mpc.fluid.gaussianRods.mdStepCount", &mdStepCount);
}

void GaussianRods::stream()
{
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(getParticleCount() / 2)
		DeviceCode::streamGaussianRodsVelocityVerlet<<<gridSize, blockSize>>>(
			workUnitOffset,
			d_mpcParticlePositions,
			d_mpcParticleVelocities,
			meanBondLength,
			springConstant / mpcParticleMass,
			streamingTimestep / mdStepCount,
			mdStepCount);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

void GaussianRods::initializeOnHost()
{
	initializeVelocitiesOnHost();

	boost::random::uniform_01<FP> posDist;

	for(unsigned int pID = 0; pID < getParticleCount(); pID+=2)
	{
		mpcParticlePositions[pID * 3 + 0] = posDist(rng) * simulation->getSimulationBoxSizeX();
		mpcParticlePositions[pID * 3 + 1] = posDist(rng) * simulation->getSimulationBoxSizeY();
		mpcParticlePositions[pID * 3 + 2] = posDist(rng) * simulation->getSimulationBoxSizeZ();

		const RemotelyStoredVector<MPCParticlePositionType> position1(mpcParticlePositions, pID);

		const Vector3D<MPCParticlePositionType> position2 = getInitialPartnerPosition(position1);

		if(	position2.getX() >= simulation->getSimulationBoxSizeX() ||
			position2.getY() >= simulation->getSimulationBoxSizeY() ||
			position2.getZ() >= simulation->getSimulationBoxSizeZ() ||
			position2.getX() < 0 || position2.getY() < 0 || position2.getZ() < 0)
		{
			//The second particle would start outside of the box,
			//thus gaining in velocity under Lees-Edwards boundary conditions.
			//Choose new positions instead.
			pID-=2;
			continue;
		}


		mpcParticlePositions[(pID+1) * 3 + 0] = position2.getX();
		mpcParticlePositions[(pID+1) * 3 + 1] = position2.getY();
		mpcParticlePositions[(pID+1) * 3 + 2] = position2.getZ();
	}
}

const Vector3D<MPCParticlePositionType>
	GaussianRods::getInitialPartnerPosition(
		const RemotelyStoredVector<MPCParticlePositionType>& position1) const
{
	const Vector3D<MPCParticlePositionType> displacement
			= Vector3D<MPCParticlePositionType>::getRandomUnitVector(rng) * meanBondLength;

	return position1 + displacement;
}
