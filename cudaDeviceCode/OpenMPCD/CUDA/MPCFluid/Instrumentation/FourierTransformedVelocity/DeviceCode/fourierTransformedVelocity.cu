#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/FourierTransformedVelocity/DeviceCode/fourierTransformedVelocity.hpp>

#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Symbols.hpp>
#include <OpenMPCD/CUDA/StridedIteratorRange.hpp>

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

const OpenMPCD::Vector3D<std::complex<OpenMPCD::MPCParticleVelocityType> >
	OpenMPCD::CUDA::MPCFluid::DeviceCode::calculateVelocityInFourierSpace_simpleMPCFluid(
		const unsigned int particleCount,
		const MPCParticlePositionType* const positions,
		const MPCParticleVelocityType* const velocities,
		const Vector3D<MPCParticlePositionType> k,
		MPCParticleVelocityType* const buffer1,
		MPCParticleVelocityType* const buffer2)
{
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(particleCount)
		calculateVelocityInFourierSpace_simpleMPCFluid_single <<<blockSize, gridSize>>> (
			workUnitOffset,
			positions,
			velocities,
			k,
			buffer1,
			buffer2);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	return reduceVelocityInFourierSpaceBuffers(particleCount, buffer1, buffer2);
}

__global__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::calculateVelocityInFourierSpace_simpleMPCFluid_single(
	const unsigned int workUnitOffset,
	const MPCParticlePositionType* const positions,
	const MPCParticleVelocityType* const velocities,
	const Vector3D<MPCParticlePositionType> k,
	MPCParticleVelocityType* const realBuffer,
	MPCParticleVelocityType* const imaginaryBuffer)
{
	const unsigned int particleID = blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;

	if(particleID >= mpcParticleCount)
		return;

	const RemotelyStoredVector<const MPCParticlePositionType> r(positions,  particleID);
	const RemotelyStoredVector<const MPCParticleVelocityType> v(velocities, particleID);

	RemotelyStoredVector<MPCParticleVelocityType> realPart     (realBuffer,      particleID);
	RemotelyStoredVector<MPCParticleVelocityType> imaginaryPart(imaginaryBuffer, particleID);

	const MPCParticleVelocityType dotProduct = k.dot(r);

	realPart      = v * cos(dotProduct);
	imaginaryPart = v * sin(dotProduct);
}









const OpenMPCD::Vector3D<std::complex<OpenMPCD::MPCParticleVelocityType> >
	OpenMPCD::CUDA::MPCFluid::DeviceCode::calculateVelocityInFourierSpace_doubletMPCFluid(
		const unsigned int doubletCount,
		const MPCParticlePositionType* const positions,
		const MPCParticleVelocityType* const velocities,
		const Vector3D<MPCParticlePositionType> k,
		MPCParticleVelocityType* const buffer1,
		MPCParticleVelocityType* const buffer2)
{
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_GRIDSIZE_BEGIN(doubletCount, 512)
		calculateVelocityInFourierSpace_doubletMPCFluid_single <<<blockSize, gridSize>>> (
			workUnitOffset,
			positions,
			velocities,
			k,
			buffer1,
			buffer2);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	return reduceVelocityInFourierSpaceBuffers(doubletCount, buffer1, buffer2);
}

__global__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::calculateVelocityInFourierSpace_doubletMPCFluid_single(
	const unsigned int workUnitOffset,
	const MPCParticlePositionType* const positions,
	const MPCParticleVelocityType* const velocities,
	const Vector3D<MPCParticlePositionType> k,
	MPCParticleVelocityType* const realBuffer,
	MPCParticleVelocityType* const imaginaryBuffer)
{
	const unsigned int doubletID = blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;
	const unsigned int particle1ID = 2 * doubletID;
	const unsigned int particle2ID = particle1ID + 1;

	if(particle2ID >= mpcParticleCount)
		return;

	const RemotelyStoredVector<const MPCParticlePositionType> r1(positions,  particle1ID);
	const RemotelyStoredVector<const MPCParticlePositionType> r2(positions,  particle2ID);

	const RemotelyStoredVector<const MPCParticleVelocityType> v1(velocities, particle1ID);
	const RemotelyStoredVector<const MPCParticleVelocityType> v2(velocities, particle2ID);


	const Vector3D<MPCParticlePositionType> r = (r1 + r2) / 2.0;
	const Vector3D<MPCParticlePositionType> v = (v1 + v2) / 2.0;


	RemotelyStoredVector<MPCParticleVelocityType> realPart     (realBuffer,      doubletID);
	RemotelyStoredVector<MPCParticleVelocityType> imaginaryPart(imaginaryBuffer, doubletID);

	const MPCParticleVelocityType dotProduct = k.dot(r);

	realPart      = v * cos(dotProduct);
	imaginaryPart = v * sin(dotProduct);
}






const OpenMPCD::Vector3D<std::complex<OpenMPCD::MPCParticleVelocityType> >
	OpenMPCD::CUDA::MPCFluid::DeviceCode::calculateVelocityInFourierSpace_tripletMPCFluid(
		const unsigned int tripletCount,
		const MPCParticlePositionType* const positions,
		const MPCParticleVelocityType* const velocities,
		const Vector3D<MPCParticlePositionType> k,
		MPCParticleVelocityType* const buffer1,
		MPCParticleVelocityType* const buffer2)
{
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(tripletCount)
		calculateVelocityInFourierSpace_tripletMPCFluid_single <<<blockSize, gridSize>>> (
			workUnitOffset,
			positions,
			velocities,
			k,
			buffer1,
			buffer2);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	return reduceVelocityInFourierSpaceBuffers(tripletCount, buffer1, buffer2);
}

__global__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::calculateVelocityInFourierSpace_tripletMPCFluid_single(
	const unsigned int workUnitOffset,
	const MPCParticlePositionType* const positions,
	const MPCParticleVelocityType* const velocities,
	const Vector3D<MPCParticlePositionType> k,
	MPCParticleVelocityType* const realBuffer,
	MPCParticleVelocityType* const imaginaryBuffer)
{
	const unsigned int tripletID = blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;
	const unsigned int particle1ID = 3 * tripletID;
	const unsigned int particle2ID = particle1ID + 1;
	const unsigned int particle3ID = particle1ID + 2;

	if(particle3ID >= mpcParticleCount)
		return;

	const RemotelyStoredVector<const MPCParticlePositionType> r1(positions,  particle1ID);
	const RemotelyStoredVector<const MPCParticlePositionType> r2(positions,  particle2ID);
	const RemotelyStoredVector<const MPCParticlePositionType> r3(positions,  particle3ID);

	const RemotelyStoredVector<const MPCParticleVelocityType> v1(velocities, particle1ID);
	const RemotelyStoredVector<const MPCParticleVelocityType> v2(velocities, particle2ID);
	const RemotelyStoredVector<const MPCParticleVelocityType> v3(velocities, particle3ID);


	const Vector3D<MPCParticlePositionType> r = (r1 + r2 + r3) / 3.0;
	const Vector3D<MPCParticlePositionType> v = (v1 + v2 + v3) / 3.0;


	RemotelyStoredVector<MPCParticleVelocityType> realPart     (realBuffer,      tripletID);
	RemotelyStoredVector<MPCParticleVelocityType> imaginaryPart(imaginaryBuffer, tripletID);

	const MPCParticleVelocityType dotProduct = k.dot(r);

	realPart      = v * cos(dotProduct);
	imaginaryPart = v * sin(dotProduct);
}






const OpenMPCD::Vector3D<std::complex<OpenMPCD::MPCParticleVelocityType> >
	OpenMPCD::CUDA::MPCFluid::DeviceCode::calculateVelocityInFourierSpace_chainMPCFluid(
		const unsigned int chainCount,
		const unsigned int chainLength,
		const MPCParticlePositionType* const positions,
		const MPCParticleVelocityType* const velocities,
		const Vector3D<MPCParticlePositionType> k,
		MPCParticleVelocityType* const buffer1,
		MPCParticleVelocityType* const buffer2)
{
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(chainCount)
		calculateVelocityInFourierSpace_chainMPCFluid_single <<<blockSize, gridSize>>> (
			workUnitOffset,
			chainLength,
			positions,
			velocities,
			k,
			buffer1,
			buffer2);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	return reduceVelocityInFourierSpaceBuffers(chainCount, buffer1, buffer2);
}

__global__ void OpenMPCD::CUDA::MPCFluid::DeviceCode::calculateVelocityInFourierSpace_chainMPCFluid_single(
	const unsigned int workUnitOffset,
	const unsigned int chainLength,
	const MPCParticlePositionType* const positions,
	const MPCParticleVelocityType* const velocities,
	const Vector3D<MPCParticlePositionType> k,
	MPCParticleVelocityType* const realBuffer,
	MPCParticleVelocityType* const imaginaryBuffer)
{
	const unsigned int chainID = blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;
	const unsigned int firstParticleID = chainID * chainLength;
	const unsigned int lastParticleID = firstParticleID + chainLength - 1;

	if(lastParticleID >= mpcParticleCount)
		return;

	Vector3D<MPCParticlePositionType> r(0, 0, 0);
	Vector3D<MPCParticleVelocityType> v(0, 0, 0);
	for(unsigned int particleID = firstParticleID; particleID <= lastParticleID; ++particleID)
	{
		const RemotelyStoredVector<const MPCParticlePositionType> r_i(positions,  particleID);
		const RemotelyStoredVector<const MPCParticleVelocityType> v_i(velocities, particleID);

		r += r_i;
		v += v_i;
	}

	r /= chainLength;
	v /= chainLength;

	RemotelyStoredVector<MPCParticleVelocityType> realPart     (realBuffer,      chainID);
	RemotelyStoredVector<MPCParticleVelocityType> imaginaryPart(imaginaryBuffer, chainID);

	const MPCParticleVelocityType dotProduct = k.dot(r);

	realPart      = v * cos(dotProduct);
	imaginaryPart = v * sin(dotProduct);
}





const OpenMPCD::Vector3D<std::complex<OpenMPCD::MPCParticleVelocityType> >
	OpenMPCD::CUDA::MPCFluid::DeviceCode::reduceVelocityInFourierSpaceBuffers(
		const unsigned int summandCount,
		MPCParticleVelocityType* const realBuffer,
		MPCParticleVelocityType* const imaginaryBuffer)
{
	const StridedIteratorRange<MPCParticleVelocityType*, 3> realValuesX(
		realBuffer + 0, realBuffer + summandCount * 3);
	const StridedIteratorRange<MPCParticleVelocityType*, 3> realValuesY(
		realBuffer + 1, realBuffer + summandCount * 3);
	const StridedIteratorRange<MPCParticleVelocityType*, 3> realValuesZ(
		realBuffer + 2, realBuffer + summandCount * 3);

	const StridedIteratorRange<MPCParticleVelocityType*, 3> imagValuesX(
		imaginaryBuffer + 0, imaginaryBuffer + summandCount * 3);
	const StridedIteratorRange<MPCParticleVelocityType*, 3> imagValuesY(
		imaginaryBuffer + 1, imaginaryBuffer + summandCount * 3);
	const StridedIteratorRange<MPCParticleVelocityType*, 3> imagValuesZ(
		imaginaryBuffer + 2, imaginaryBuffer + summandCount * 3);


	const MPCParticleVelocityType realPartX
		= thrust::reduce(thrust::device, realValuesX.begin(), realValuesX.end()) / summandCount;

	const MPCParticleVelocityType realPartY
		= thrust::reduce(thrust::device, realValuesY.begin(), realValuesY.end()) / summandCount;

	const MPCParticleVelocityType realPartZ
		= thrust::reduce(thrust::device, realValuesZ.begin(), realValuesZ.end()) / summandCount;


	const MPCParticleVelocityType imagPartX
		= thrust::reduce(thrust::device, imagValuesX.begin(), imagValuesX.end()) / summandCount;

	const MPCParticleVelocityType imagPartY
		= thrust::reduce(thrust::device, imagValuesY.begin(), imagValuesY.end()) / summandCount;

	const MPCParticleVelocityType imagPartZ
		= thrust::reduce(thrust::device, imagValuesZ.begin(), imagValuesZ.end()) / summandCount;


	const std::complex<MPCParticleVelocityType> x(realPartX, imagPartX);
	const std::complex<MPCParticleVelocityType> y(realPartY, imagPartY);
	const std::complex<MPCParticleVelocityType> z(realPartZ, imagPartZ);

	return Vector3D<std::complex<MPCParticleVelocityType> >(x, y, z);
}
