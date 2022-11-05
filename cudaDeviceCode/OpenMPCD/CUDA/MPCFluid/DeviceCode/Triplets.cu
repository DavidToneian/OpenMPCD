#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Triplets.hpp>

#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Symbols.hpp>

void OpenMPCD::CUDA::MPCFluid::DeviceCode::getCenterOfMassVelocities_triplet(
	const unsigned int mpcParticleCount,
	const MPCParticleVelocityType* const velocities,
	MPCParticleVelocityType* const comVelocities)
{
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(mpcParticleCount / 3)
		getCenterOfMassVelocities_triplet_kernel<<<gridSize, blockSize>>>(
			workUnitOffset,
			velocities,
			comVelocities);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

__global__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::getCenterOfMassVelocities_triplet_kernel(
	const unsigned int workUnitOffset,
	const MPCParticleVelocityType* const velocities,
	MPCParticleVelocityType* const comVelocities)
{
	const unsigned int tripletID = blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;

	if(tripletID * 3 + 2 >= mpcParticleCount)
		return;

	RemotelyStoredVector<MPCParticleVelocityType> result(comVelocities, tripletID);

	result = getCenterOfMassVelocity_doublet(tripletID, velocities);
}

__device__ OpenMPCD::Vector3D<OpenMPCD::MPCParticleVelocityType>
	OpenMPCD::CUDA::MPCFluid::DeviceCode::getCenterOfMassVelocity_triplet(
			const unsigned int tripletID,
			const MPCParticleVelocityType* const velocities)
{
	const RemotelyStoredVector<const MPCParticleVelocityType> v_1(velocities, tripletID * 3 + 0);
	const RemotelyStoredVector<const MPCParticleVelocityType> v_2(velocities, tripletID * 3 + 1);
	const RemotelyStoredVector<const MPCParticleVelocityType> v_3(velocities, tripletID * 3 + 2);

	return (v_1 + v_2 + v_3) / 3.0;
}


