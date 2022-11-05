#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Chains.hpp>

#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Symbols.hpp>

void OpenMPCD::CUDA::MPCFluid::DeviceCode::getCenterOfMassVelocities_chain(
	const unsigned int mpcParticleCount,
	const unsigned int chainLength,
	const MPCParticleVelocityType* const velocities,
	MPCParticleVelocityType* const comVelocities)
{
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(mpcParticleCount / chainLength)
		getCenterOfMassVelocities_chain_kernel<<<gridSize, blockSize>>>(
			workUnitOffset,
			chainLength,
			velocities,
			comVelocities);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

__global__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::getCenterOfMassVelocities_chain_kernel(
	const unsigned int workUnitOffset,
	const unsigned int chainLength,
	const MPCParticleVelocityType* const velocities,
	MPCParticleVelocityType* const comVelocities)
{
	const unsigned int chainID = blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;

	const unsigned int chainCount = mpcParticleCount / chainLength;

	if(chainID >= chainCount)
		return;

	RemotelyStoredVector<MPCParticleVelocityType> result(comVelocities, chainID);

	result = getCenterOfMassVelocity_chain(chainID, chainLength, velocities);
}

__device__ OpenMPCD::Vector3D<OpenMPCD::MPCParticleVelocityType>
	OpenMPCD::CUDA::MPCFluid::DeviceCode::getCenterOfMassVelocity_chain(
			const unsigned int chainID,
			const unsigned int chainLength,
			const MPCParticleVelocityType* const velocities)
{
	Vector3D<MPCParticleVelocityType> v(0, 0, 0);

	const unsigned int firstParticleID = chainID * chainLength;

	for(unsigned int particleOffset = 0; particleOffset < chainLength; ++particleOffset)
	{
		const RemotelyStoredVector<const MPCParticleVelocityType> v_i(velocities, firstParticleID + particleOffset);

		v += v_i;
	}

	return v / chainLength;
}


