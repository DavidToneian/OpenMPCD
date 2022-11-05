#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/GaussianDumbbells.hpp>

#include <OpenMPCD/CUDA/DeviceCode/LeesEdwardsBoundaryConditions.hpp>
#include <OpenMPCD/CUDA/DeviceCode/Simulation.hpp>
#include <OpenMPCD/CUDA/DeviceCode/Symbols.hpp>
#include <OpenMPCD/CUDA/DeviceCode/VelocityVerlet.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Symbols.hpp>

void OpenMPCD::CUDA::MPCFluid::DeviceCode::setGaussianDumbbellSymbols(
	const FP omega_, const FP timestep)
{
	cudaMemcpyToSymbol(omega, &omega_, sizeof(omega_));
	OPENMPCD_CUDA_THROW_ON_ERROR;

	const FP c = cos(omega_*timestep);
	const FP s = sin(omega_*timestep);

	cudaMemcpyToSymbol(cos_omegaTimesTimestep, &c, sizeof(c));
	OPENMPCD_CUDA_THROW_ON_ERROR;

	cudaMemcpyToSymbol(sin_omegaTimesTimestep, &s, sizeof(s));
	OPENMPCD_CUDA_THROW_ON_ERROR;
}


__device__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::streamDumbbellAnalytically(
	const unsigned int particle1ID, OpenMPCD::MPCParticlePositionType* const positions,
	OpenMPCD::MPCParticleVelocityType* const velocities)
{
	using namespace OpenMPCD::CUDA::DeviceCode;

	#ifdef OPENMPCD_CUDA_DEBUG
		if(particle1ID + 1 >= mpcParticleCount)
			printf("Bad index particle1ID in OpenMPCD::CUDA::MPCFluid::DeviceCode::streamDumbbellAnalytically\n");
	#endif

	RemotelyStoredVector<MPCParticlePositionType> r_1(positions, particle1ID);
	RemotelyStoredVector<MPCParticlePositionType> r_2(positions, particle1ID + 1);

	RemotelyStoredVector<MPCParticleVelocityType> v_1(velocities, particle1ID);
	RemotelyStoredVector<MPCParticleVelocityType> v_2(velocities, particle1ID + 1);

	#ifdef OPENMPCD_CUDA_DEBUG_STREAMING
		if(particle1ID == 0)
		{
			printf("Before streaming:\nPositions: %g %g %g\nVelocities: %g %g %g\n",
				r_1.getX(), r_1.getY(), r_1.getZ(),
				v_1.getX(), v_1.getY(), v_1.getZ());
		}
	#endif

	#ifdef OPENMPCD_CUDA_DEBUG
		if(!r_1.isFinite())
			printf(	"r_1 at the beginning of OpenMPCD::CUDA::MPCFluid::DeviceCode::streamDumbbellAnalytically is not finite. "
					"Values: %g %g %g\n", r_1.getX(), r_1.getY(), r_1.getZ());
		if(!r_2.isFinite())
			printf(	"r_2 at the beginning of OpenMPCD::CUDA::MPCFluid::DeviceCode::streamDumbbellAnalytically is not finite. "
					"Values: %g %g %g\n", r_2.getX(), r_2.getY(), r_2.getZ());

		if(!v_1.isFinite())
			printf(	"v_1 at the beginning of OpenMPCD::CUDA::MPCFluid::DeviceCode::streamDumbbellAnalytically is not finite. "
					"Values: %g %g %g\n", v_1.getX(), v_1.getY(), v_1.getZ());
		if(!v_2.isFinite())
			printf(	"v_2 at the beginning of OpenMPCD::CUDA::MPCFluid::DeviceCode::streamDumbbellAnalytically is not finite. "
					"Values: %g %g %g\n", v_2.getX(), v_2.getY(), v_2.getZ());
	#endif

	Vector3D<MPCParticlePositionType> r_cm = (r_1 + r_2) / 2.0;
	Vector3D<MPCParticlePositionType> R    = r_2 - r_1;

	Vector3D<MPCParticleVelocityType> v_cm = (v_1 + v_2) / 2.0;
	Vector3D<MPCParticleVelocityType> v_R  = v_2 - v_1;


	//with acceleration:
	//r_cm += v_cm * d_streamingTimestep + acceleration * d_streamingTimestep * d_streamingTimestep / 2.0;
	//v_cm += acceleration * d_streamingTimestep;

	//without acceleration:
	r_cm += v_cm * CUDA::DeviceCode::streamingTimestep;

	const Vector3D<MPCParticlePositionType> R_old = R;
	R   = R * cos_omegaTimesTimestep + v_R * sin_omegaTimesTimestep / omega;
	v_R = - omega * R_old * sin_omegaTimesTimestep + v_R * cos_omegaTimesTimestep;

	R   /= 2.0;
	v_R /= 2.0;

	r_1 = r_cm - R;
	r_2 = r_cm + R;

	v_1 = v_cm - v_R;
	v_2 = v_cm + v_R;

	#ifdef OPENMPCD_CUDA_DEBUG_STREAMING
		if(particle1ID == 0)
		{
			printf("After streaming:\nPositions: %g %g %g\nVelocities: %g %g %g\n",
				r_1.getX(), r_1.getY(), r_1.getZ(),
				v_1.getX(), v_1.getY(), v_1.getZ());
		}
	#endif

	#ifdef OPENMPCD_CUDA_DEBUG
		if(!r_1.isFinite())
			printf(	"r_1 at the end of OpenMPCD::CUDA::MPCFluid::DeviceCode::streamDumbbellAnalytically is not finite. "
					"Values: %g %g %g\n", r_1.getX(), r_1.getY(), r_1.getZ());
		if(!r_2.isFinite())
			printf(	"r_2 at the end of OpenMPCD::CUDA::MPCFluid::DeviceCode::streamDumbbellAnalytically is not finite. "
					"Values: %g %g %g\n", r_2.getX(), r_2.getY(), r_2.getZ());

		if(!v_1.isFinite())
			printf(	"v_1 at the end of OpenMPCD::CUDA::MPCFluid::DeviceCode::streamDumbbellAnalytically is not finite. "
					"Values: %g %g %g\n", v_1.getX(), v_1.getY(), v_1.getZ());
		if(!v_2.isFinite())
			printf(	"v_2 at the end of OpenMPCD::CUDA::MPCFluid::DeviceCode::streamDumbbellAnalytically is not finite. "
					"Values: %g %g %g\n", v_2.getX(), v_2.getY(), v_2.getZ());
	#endif
}

__global__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::streamDumbbellsAnalytically(
	const unsigned int workUnitOffset,
	MPCParticlePositionType* const positions,
	MPCParticleVelocityType* const velocities)
{
	using namespace OpenMPCD::CUDA::DeviceCode;

	const unsigned int particleID = 2 * (blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset);

	if(particleID >= mpcParticleCount)
		return;

	streamDumbbellAnalytically(particleID, positions, velocities);
}

__device__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::streamDumbbellVelocityVerlet(
	const unsigned int particle1ID,
	OpenMPCD::MPCParticlePositionType* const positions,
	OpenMPCD::MPCParticleVelocityType* const velocities,
	const FP reducedSpringConstant,
	const FP timestep,
	const unsigned int stepCount)
{
	using namespace OpenMPCD::CUDA::DeviceCode;

	#ifdef OPENMPCD_CUDA_DEBUG
		if(particle1ID + 1 >= mpcParticleCount)
		{
			printf("Bad index particle1ID in OpenMPCD::CUDA::MPCFluid::DeviceCode::streamDumbbellVelocityVerlet\n");
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

		const Vector3D<FP> a_1_old = reducedSpringConstant * R;
		const Vector3D<FP> a_2_old = - a_1_old;

		OpenMPCD::CUDA::DeviceCode::velocityVerletStep1(&r_1, v_1, a_1_old, timestep);
		OpenMPCD::CUDA::DeviceCode::velocityVerletStep1(&r_2, v_2, a_2_old, timestep);

		R = r_2 - r_1;

		const Vector3D<FP> a_1_new = reducedSpringConstant * R;
		const Vector3D<FP> a_2_new = - a_1_new;

		OpenMPCD::CUDA::DeviceCode::velocityVerletStep2(&v_1, a_1_old, a_1_new, timestep);
		OpenMPCD::CUDA::DeviceCode::velocityVerletStep2(&v_2, a_2_old, a_2_new, timestep);
	}
}

__global__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::streamDumbbellsVelocityVerlet(
	const unsigned int workUnitOffset,
	MPCParticlePositionType* const positions,
	MPCParticleVelocityType* const velocities,
	const FP reducedSpringConstant,
	const FP timestep,
	const unsigned int stepCount)
{
	using namespace OpenMPCD::CUDA::DeviceCode;

	const unsigned int particleID = 2 * (blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset);

	if(particleID >= mpcParticleCount)
		return;

	streamDumbbellVelocityVerlet(particleID, positions, velocities, reducedSpringConstant, timestep, stepCount);
}
