#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/GaussianRods.hpp>

#include <OpenMPCD/CUDA/DeviceCode/LeesEdwardsBoundaryConditions.hpp>
#include <OpenMPCD/CUDA/DeviceCode/Simulation.hpp>
#include <OpenMPCD/CUDA/DeviceCode/VelocityVerlet.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>

__device__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::streamGaussianRodVelocityVerlet(
	const unsigned int particle1ID,
	OpenMPCD::MPCParticlePositionType* const positions,
	OpenMPCD::MPCParticleVelocityType* const velocities,
	const FP meanBondLength,
	const FP reducedSpringConstant,
	const FP timestep,
	const unsigned int stepCount)
{
	using namespace OpenMPCD::CUDA::DeviceCode;

	#ifdef OPENMPCD_CUDA_DEBUG
		if(particle1ID + 1 >= mpcParticleCount)
		{
			printf("Bad index particle1ID in OpenMPCD::CUDA::MPCFluid::DeviceCode::streamGaussianRodVelocityVerlet\n");
			assert(0);
		}
	#endif

	RemotelyStoredVector<MPCParticlePositionType> r_1(positions, particle1ID);
	RemotelyStoredVector<MPCParticlePositionType> r_2(positions, particle1ID + 1);

	RemotelyStoredVector<MPCParticleVelocityType> v_1(velocities, particle1ID);
	RemotelyStoredVector<MPCParticleVelocityType> v_2(velocities, particle1ID + 1);

	for(unsigned int step=0; step<stepCount; ++step)
	{
		Vector3D<MPCParticlePositionType> R = r_2 - r_1;
		MPCParticlePositionType R_magnitude = R.getMagnitude();

		if(R_magnitude == 0)
		{
			printf("Bond length of zero in OpenMPCD::CUDA::MPCFluid::DeviceCode::streamGaussianRodVelocityVerlet\n");
			assert(0);
		}

		const Vector3D<FP> a_1_old = reducedSpringConstant * (1 - meanBondLength / R_magnitude) * R;
		const Vector3D<FP> a_2_old = - a_1_old;

		OpenMPCD::CUDA::DeviceCode::velocityVerletStep1(&r_1, v_1, a_1_old, timestep);
		OpenMPCD::CUDA::DeviceCode::velocityVerletStep1(&r_2, v_2, a_2_old, timestep);

		R = r_2 - r_1;
		R_magnitude = R.getMagnitude();

		if(R_magnitude == 0)
		{
			printf("Bond length of zero in OpenMPCD::CUDA::MPCFluid::DeviceCode::streamGaussianRodVelocityVerlet\n");
			assert(0);
		}

		const Vector3D<FP> a_1_new = reducedSpringConstant * (1 - meanBondLength / R_magnitude) * R;
		const Vector3D<FP> a_2_new = - a_1_new;

		OpenMPCD::CUDA::DeviceCode::velocityVerletStep2(&v_1, a_1_old, a_1_new, timestep);
		OpenMPCD::CUDA::DeviceCode::velocityVerletStep2(&v_2, a_2_old, a_2_new, timestep);
	}
}

__global__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::streamGaussianRodsVelocityVerlet(
	const unsigned int workUnitOffset,
	MPCParticlePositionType* const positions,
	MPCParticleVelocityType* const velocities,
	const FP meanBondLength,
	const FP reducedSpringConstant,
	const FP timestep,
	const unsigned int stepCount)
{
	using namespace OpenMPCD::CUDA::DeviceCode;

	const unsigned int particleID = 2 * (blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset);

	if(particleID >= mpcParticleCount)
		return;

	streamGaussianRodVelocityVerlet(
		particleID, positions, velocities, meanBondLength, reducedSpringConstant, timestep, stepCount);
}
