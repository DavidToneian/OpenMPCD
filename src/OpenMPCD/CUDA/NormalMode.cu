#include <OpenMPCD/CUDA/NormalMode.hpp>

#include <OpenMPCD/CUDA/DeviceCode/NormalMode.hpp>
#include <OpenMPCD/CUDA/BunchIteratorRange.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>

namespace OpenMPCD
{
namespace CUDA
{
namespace NormalMode
{

void computeNormalCoordinates(
	const unsigned int chainLength,
	const unsigned int chainCount,
	const MPCParticlePositionType* const positions,
	MPCParticlePositionType* const normalModeCoordinates,
	const MPCParticlePositionType shift)
{
	typedef MPCParticlePositionType T;

	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		chainLength != 0, InvalidArgumentException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		positions, NULLPointerException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		normalModeCoordinates, NULLPointerException);

	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(chainCount)
		OpenMPCD::CUDA::DeviceCode::NormalMode::
		computeNormalCoordinates<T> <<<gridSize, blockSize>>>(
			workUnitOffset,
			chainLength,
			chainCount,
			positions,
			normalModeCoordinates,
			shift);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

const std::vector<MPCParticlePositionType>
getAverageNormalCoordinateAutocorrelation(
	const unsigned int chainLength,
	const unsigned int chainCount,
	const MPCParticlePositionType* const normalModes0,
	const MPCParticlePositionType* const normalModesT)
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		chainLength != 0, InvalidArgumentException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(normalModes0, NULLPointerException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(normalModesT, NULLPointerException);


	std::vector<MPCParticlePositionType> ret;
	ret.reserve(chainLength + 1);

	typedef thrust::device_ptr<const MPCParticleVelocityType> DevicePointer;

	const DevicePointer d_normalModes0 =
		thrust::device_pointer_cast(normalModes0);
	const DevicePointer d_normalModesT =
		thrust::device_pointer_cast(normalModesT);

	for(unsigned int mode = 0; mode <= chainLength; ++mode)
	{
		const unsigned int gapSize = 3 * chainLength;
		const unsigned int offset = 3 * mode;

		const BunchIteratorRange<DevicePointer> range0(
			d_normalModes0 + offset,
			d_normalModes0 + 3 * chainCount * (chainLength + 1),
			3,
			gapSize);
		const BunchIteratorRange<DevicePointer> rangeT(
			d_normalModesT + offset,
			d_normalModesT + 3 * chainCount * (chainLength + 1),
			3,
			gapSize);


		const MPCParticlePositionType autocorrelationSum =
			thrust::inner_product(
				thrust::device,
				range0.begin(),
				range0.end(),
				rangeT.begin(),
				MPCParticlePositionType(0));

		const MPCParticleVelocityType autocorrelation =
			autocorrelationSum / chainCount;

		ret.push_back(autocorrelation);
	}

	return ret;
}

} //namespace NormalMode
} //namespace CUDA
} //namespace OpenMPCD
