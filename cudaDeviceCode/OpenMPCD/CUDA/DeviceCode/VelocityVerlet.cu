#include <OpenMPCD/CUDA/DeviceCode/VelocityVerlet.hpp>

__device__ OpenMPCD::MPCParticlePositionType
	OpenMPCD::CUDA::DeviceCode::velocityVerletStep1(
		const MPCParticlePositionType position,
		const MPCParticleVelocityType velocity,
		const FP acceleration, const FP timestep)
{
	return position + velocity * timestep + 0.5 * acceleration * timestep * timestep;
}

__device__ OpenMPCD::MPCParticleVelocityType
	OpenMPCD::CUDA::DeviceCode::velocityVerletStep2(
		const MPCParticleVelocityType velocity,
		const FP oldAcceleration,
		const FP newAcceleration,
		const FP timestep)
{
	return velocity + 0.5 * timestep * (oldAcceleration + newAcceleration);
}

__device__ void
	OpenMPCD::CUDA::DeviceCode::velocityVerletStep1(
		RemotelyStoredVector<MPCParticlePositionType>* const position,
		const RemotelyStoredVector<MPCParticleVelocityType> velocity,
		const Vector3D<FP> acceleration, const FP timestep)
{
	#ifdef OPENMPCD_CUDA_DEBUG
		if(position==NULL)
			printf("NULL pointer given in OpenMPCD::CUDA::DeviceCode::velocityVerletStep1");
	#endif

	position->setX(velocityVerletStep1(position->getX(), velocity.getX(), acceleration.getX(), timestep));
	position->setY(velocityVerletStep1(position->getY(), velocity.getY(), acceleration.getY(), timestep));
	position->setZ(velocityVerletStep1(position->getZ(), velocity.getZ(), acceleration.getZ(), timestep));
}

__device__ void
	OpenMPCD::CUDA::DeviceCode::velocityVerletStep2(
		RemotelyStoredVector<MPCParticleVelocityType>* const velocity,
		const Vector3D<FP> oldAcceleration,
		const Vector3D<FP> newAcceleration,
		const FP timestep)
{
	#ifdef OPENMPCD_CUDA_DEBUG
		if(velocity==NULL)
			printf("NULL pointer given in OpenMPCD::CUDA::DeviceCode::velocityVerletStep2");
	#endif

	velocity->setX(velocityVerletStep2(velocity->getX(), oldAcceleration.getX(), newAcceleration.getX(), timestep));
	velocity->setY(velocityVerletStep2(velocity->getY(), oldAcceleration.getY(), newAcceleration.getY(), timestep));
	velocity->setZ(velocityVerletStep2(velocity->getZ(), oldAcceleration.getZ(), newAcceleration.getZ(), timestep));
}
