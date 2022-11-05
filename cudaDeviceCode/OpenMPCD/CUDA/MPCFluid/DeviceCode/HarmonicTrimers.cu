#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/HarmonicTrimers.hpp>

#include <OpenMPCD/CUDA/DeviceCode/VelocityVerlet.hpp>

__global__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::streamHarmonicTrimerVelocityVerlet(
	const unsigned int workUnitOffset,
	MPCParticlePositionType* const positions,
	MPCParticleVelocityType* const velocities,
	const FP reducedSpringConstant1,
	const FP reducedSpringConstant2,
	const FP timestep,
	const unsigned int stepCount)
{
	const FP k_1 = reducedSpringConstant1;
	const FP k_2 = reducedSpringConstant2;

	const unsigned int particle1ID = 3 * (blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset);

	if(particle1ID + 2 >= mpcParticleCount)
		return;

	RemotelyStoredVector<MPCParticlePositionType> r_1(positions,  particle1ID + 0);
	RemotelyStoredVector<MPCParticleVelocityType> v_1(velocities, particle1ID + 0);

	RemotelyStoredVector<MPCParticlePositionType> r_2(positions,  particle1ID + 1);
	RemotelyStoredVector<MPCParticleVelocityType> v_2(velocities, particle1ID + 1);

	RemotelyStoredVector<MPCParticlePositionType> r_3(positions,  particle1ID + 2);
	RemotelyStoredVector<MPCParticleVelocityType> v_3(velocities, particle1ID + 2);

	for(unsigned int step=0; step<stepCount; ++step)
	{
		Vector3D<MPCParticlePositionType> R_1 = r_2 - r_1;
		Vector3D<MPCParticlePositionType> R_2 = r_3 - r_2;

		const Vector3D<FP> a_1_old = k_1 * R_1;
		const Vector3D<FP> a_2_old = - k_1 * R_1 + k_2 * R_2;
		const Vector3D<FP> a_3_old = - k_2 * R_2;

		OpenMPCD::CUDA::DeviceCode::velocityVerletStep1(&r_1, v_1, a_1_old, timestep);
		OpenMPCD::CUDA::DeviceCode::velocityVerletStep1(&r_2, v_2, a_2_old, timestep);
		OpenMPCD::CUDA::DeviceCode::velocityVerletStep1(&r_3, v_3, a_3_old, timestep);



		R_1 = r_2 - r_1;
		R_2 = r_3 - r_2;

		const Vector3D<FP> a_1_new = k_1 * R_1;
		const Vector3D<FP> a_2_new = - k_1 * R_1 + k_2 * R_2;
		const Vector3D<FP> a_3_new = - k_2 * R_2;

		OpenMPCD::CUDA::DeviceCode::velocityVerletStep2(&v_1, a_1_old, a_1_new, timestep);
		OpenMPCD::CUDA::DeviceCode::velocityVerletStep2(&v_2, a_2_old, a_2_new, timestep);
		OpenMPCD::CUDA::DeviceCode::velocityVerletStep2(&v_3, a_3_old, a_3_new, timestep);
	}
}
