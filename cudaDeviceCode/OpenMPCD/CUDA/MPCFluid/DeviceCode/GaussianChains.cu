#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/GaussianChains.hpp>

#include <OpenMPCD/CUDA/DeviceCode/LeesEdwardsBoundaryConditions.hpp>
#include <OpenMPCD/CUDA/DeviceCode/Simulation.hpp>
#include <OpenMPCD/CUDA/DeviceCode/VelocityVerlet.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Symbols.hpp>


__device__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::streamGaussianChainVelocityVerlet(
	const unsigned int particle1ID,
	MPCParticlePositionType* const positions,
	MPCParticleVelocityType* const velocities,
	FP* const accelerationBuffer,
	const unsigned int particlesPerChain,
	const FP reducedSpringConstant,
	const FP timestep,
	const unsigned int stepCount)
{
	using namespace OpenMPCD::CUDA::DeviceCode;

	#ifdef OPENMPCD_CUDA_DEBUG
		if(particle1ID + particlesPerChain - 1 >= mpcParticleCount)
		{
			printf("Bad index particle1ID in OpenMPCD::CUDA::MPCFluid::DeviceCode::streamGaussianChainVelocityVerlet\n");
			printf("particle1ID: %u\n", particle1ID);
			printf("particlesPerChain: %u\n", particlesPerChain);
			printf("mpcParticleCount: %u\n", mpcParticleCount);
			assert(0);
		}
	#endif

	const unsigned int lastParticleID = particle1ID + particlesPerChain - 1;

	for(unsigned int step = 0; step < stepCount; ++step)
	{
		for(unsigned int partnerOffset = 0; partnerOffset < particlesPerChain; ++partnerOffset)
		{
			const unsigned int currentParticleID = particle1ID + partnerOffset;

			RemotelyStoredVector<FP> oldAcceleration(accelerationBuffer, currentParticleID);
			oldAcceleration
				= getAccelerationGaussianChainVelocityVerlet(
					positions, particle1ID, currentParticleID, lastParticleID, reducedSpringConstant, timestep);
		}

		for(unsigned int partnerOffset = 0; partnerOffset < particlesPerChain; ++partnerOffset)
		{
			const unsigned int currentParticleID = particle1ID + partnerOffset;

			RemotelyStoredVector<MPCParticlePositionType> r(positions, currentParticleID);
			const RemotelyStoredVector<MPCParticleVelocityType> v(velocities, currentParticleID);
			const RemotelyStoredVector<FP> oldAcceleration(accelerationBuffer, currentParticleID);

			OpenMPCD::CUDA::DeviceCode::velocityVerletStep1(&r, v, oldAcceleration, timestep);
		}

		for(unsigned int partnerOffset = 0; partnerOffset < particlesPerChain; ++partnerOffset)
		{
			const unsigned int currentParticleID = particle1ID + partnerOffset;

			RemotelyStoredVector<MPCParticleVelocityType> v(velocities, currentParticleID);
			const RemotelyStoredVector<FP> oldAcceleration(accelerationBuffer, currentParticleID);
			const Vector3D<FP> newAcceleration
				= getAccelerationGaussianChainVelocityVerlet(
					positions, particle1ID, currentParticleID, lastParticleID, reducedSpringConstant, timestep);

			OpenMPCD::CUDA::DeviceCode::velocityVerletStep2(&v, oldAcceleration, newAcceleration, timestep);
		}
	}
}

__global__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::streamGaussianChainsVelocityVerlet(
	const unsigned int workUnitOffset,
	MPCParticlePositionType* const positions,
	MPCParticleVelocityType* const velocities,
	FP* const accelerationBuffer,
	const unsigned int particlesPerChain,
	const FP reducedSpringConstant,
	const FP timestep,
	const unsigned int stepCount)
{
	using namespace OpenMPCD::CUDA::DeviceCode;

	const unsigned int particleID = particlesPerChain * (blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset);

	if(particleID >= mpcParticleCount)
		return;

	streamGaussianChainVelocityVerlet(
		particleID, positions, velocities, accelerationBuffer,
		particlesPerChain, reducedSpringConstant, timestep, stepCount);
}

__device__ const OpenMPCD::Vector3D<OpenMPCD::FP>
	OpenMPCD::CUDA::MPCFluid::DeviceCode::getAccelerationGaussianChainVelocityVerlet(
		MPCParticlePositionType* const positions,
		const unsigned int firstParticleID,
		const unsigned int particleID,
		const unsigned int lastParticleID,
		const FP reducedSpringConstant,
		const FP timestep)
{
	Vector3D<FP> accelerationDueToPrevious(0, 0, 0);
	Vector3D<FP> accelerationDueToNext(0, 0, 0);

	RemotelyStoredVector<MPCParticlePositionType> r(positions, particleID);

	if(particleID != firstParticleID)
	{
		RemotelyStoredVector<MPCParticlePositionType> r_previous(positions, particleID - 1);
		const Vector3D<MPCParticlePositionType> R = r - r_previous;

		accelerationDueToPrevious = - reducedSpringConstant * R;
	}

	if(particleID != lastParticleID)
	{
		RemotelyStoredVector<MPCParticlePositionType> r_next(positions, particleID + 1);
		const Vector3D<MPCParticlePositionType> R = r_next - r;

		accelerationDueToNext = reducedSpringConstant * R;
	}

	return accelerationDueToPrevious + accelerationDueToNext;
}
