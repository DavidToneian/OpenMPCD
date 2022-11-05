#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Doublets.hpp>

#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Symbols.hpp>

void OpenMPCD::CUDA::MPCFluid::DeviceCode::getCenterOfMassVelocities_doublet(
	const unsigned int mpcParticleCount,
	const MPCParticleVelocityType* const velocities,
	MPCParticleVelocityType* const comVelocities)
{
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(mpcParticleCount / 2)
		getCenterOfMassVelocities_doublet_kernel<<<gridSize, blockSize>>>(
			workUnitOffset,
			velocities,
			comVelocities);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

__global__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::getCenterOfMassVelocities_doublet_kernel(
	const unsigned int workUnitOffset,
	const MPCParticleVelocityType* const velocities,
	MPCParticleVelocityType* const comVelocities)
{
	const unsigned int doubletID = blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;

	if(doubletID * 2 + 1 >= mpcParticleCount)
		return;

	RemotelyStoredVector<MPCParticleVelocityType> result(comVelocities, doubletID);

	result = getCenterOfMassVelocity_doublet(doubletID, velocities);
}

__device__ OpenMPCD::Vector3D<OpenMPCD::MPCParticleVelocityType>
	OpenMPCD::CUDA::MPCFluid::DeviceCode::getCenterOfMassVelocity_doublet(
			const unsigned int doubletID,
			const MPCParticleVelocityType* const velocities)
{
	const RemotelyStoredVector<const MPCParticleVelocityType> v_1(velocities, doubletID * 2 + 0);
	const RemotelyStoredVector<const MPCParticleVelocityType> v_2(velocities, doubletID * 2 + 1);

	return (v_1 + v_2) / 2.0;
}


