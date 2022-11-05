/**
 * @file
 * Holds implementation details for `OpenMPCD/CUDA/DeviceCode/Average.hpp`.
 */

#ifndef OPENMPCD_CUDA_DEVICECODE_IMPLEMENTATIONDETAILS_AVERAGE
#define OPENMPCD_CUDA_DEVICECODE_IMPLEMENTATIONDETAILS_AVERAGE

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

namespace OpenMPCD
{
namespace CUDA
{
namespace DeviceCode
{

template<typename T>
	__global__ void arithmeticMean_kernel(
		const T* const values,
		const std::size_t numberOfValues,
		T* const output)
{
	const T result =
		thrust::reduce(thrust::device, values, values + numberOfValues);
	*output = result / numberOfValues;
}

template<typename T>
	void arithmeticMean(
		const T* const values,
		const std::size_t numberOfValues,
		T* const output)
{
	#ifdef OPENMPCD_DEBUG
		if(!values)
			OPENMPCD_THROW(NULLPointerException, "`values`");

		if(numberOfValues == 0)
			OPENMPCD_THROW(InvalidArgumentException, "`numberOfValues`");

		if(!output)
			OPENMPCD_THROW(NULLPointerException, "`output`");

		if(!DeviceMemoryManager::isDeviceMemoryPointer(values))
			OPENMPCD_THROW(InvalidArgumentException, "`values`");

		if(!DeviceMemoryManager::isDeviceMemoryPointer(output))
			OPENMPCD_THROW(InvalidArgumentException, "`output`");
	#endif

	const T result =
		thrust::reduce(thrust::device, values, values + numberOfValues) /
		numberOfValues;

	DeviceMemoryManager::copyElementsFromHostToDevice(&result, output, 1);
}

} //namespace DeviceCode
} //namespace CUDA
} //namespace OpenMPCD

#endif //OPENMPCD_CUDA_DEVICECODE_IMPLEMENTATIONDETAILS_AVERAGE
