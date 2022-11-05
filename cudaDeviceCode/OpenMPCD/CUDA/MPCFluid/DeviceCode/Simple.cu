#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Simple.hpp>

#include <OpenMPCD/CUDA/DeviceCode/Symbols.hpp>
#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Symbols.hpp>

__global__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::streamSimpleMPCParticle(
	const unsigned int workUnitOffset,
	MPCParticlePositionType* const positions,
	const MPCParticleVelocityType* const velocities)
{
	const unsigned int particleID = blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;

	if(particleID >= mpcParticleCount)
		return;

	RemotelyStoredVector<MPCParticlePositionType> r(positions, particleID);
	const RemotelyStoredVector<const MPCParticleVelocityType> v(velocities, particleID);

	#ifdef OPENMPCD_CUDA_DEBUG
		if(!r.isFinite() || !v.isFinite())
		{
			printf(
				"%s\n%s %u\n%s %g %g %g\n%s %g %g %g\n",
				"Bad positions/velocities given in OpenMPCD::CUDA::MPCFluid::DeviceCode::streamSimpleMPCParticle.",
				"Particle ID:", particleID,
				"Position:", r.getX(), r.getY(), r.getZ(),
				"Velocity:", v.getX(), v.getY(), v.getZ()
				);
		}
	#endif

	r += v * CUDA::DeviceCode::streamingTimestep;

	#ifdef OPENMPCD_CUDA_DEBUG
		if(!r.isFinite())
		{
			printf(
				"%s\n%s %u\n%s %g %g %g\n%s %g %g %g\n",
				"Bad positions calculated in OpenMPCD::CUDA::MPCFluid::DeviceCode::streamSimpleMPCParticle.",
				"Particle ID:", particleID,
				"Position:", r.getX(), r.getY(), r.getZ(),
				"Velocity:", v.getX(), v.getY(), v.getZ()
				);
		}
	#endif
}
