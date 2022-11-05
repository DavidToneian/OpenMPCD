/**
 * @file
 * Implements some of the functionality of `OpenMPCD::CUDA::DeviceMemoryManager`.
 */

#ifndef OPENMPCD_CUDA_IMPLEMENTATIONDETAILS_DEVICEMEMORYMANAGER_HPP
#define OPENMPCD_CUDA_IMPLEMENTATIONDETAILS_DEVICEMEMORYMANAGER_HPP

#include <OpenMPCD/CUDA/runtime.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>

#include <boost/scoped_array.hpp>

#include <string.h>

namespace OpenMPCD
{
namespace CUDA
{

template<typename T>
	void DeviceMemoryManager::copyElementsFromHostToDevice(
		const T* const src, T* const dest, const std::size_t count)
{
	#ifdef OPENMPCD_DEBUG
		if(!src)
			OPENMPCD_THROW(NULLPointerException, "`src`");

		if(!dest)
			OPENMPCD_THROW(NULLPointerException, "`dest`");

		if(!isHostMemoryPointer(src))
			OPENMPCD_THROW(InvalidArgumentException, "`src`");

		if(!isDeviceMemoryPointer(dest))
			OPENMPCD_THROW(InvalidArgumentException, "`dest`");
	#endif

	cudaMemcpy(dest, src, sizeof(T) * count, cudaMemcpyHostToDevice);
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

template<typename T>
	void DeviceMemoryManager::copyElementsFromDeviceToHost(
		const T* const src, T* const dest, const std::size_t count)
{
	#ifdef OPENMPCD_DEBUG
		if(!src)
			OPENMPCD_THROW(NULLPointerException, "`src`");

		if(!dest)
			OPENMPCD_THROW(NULLPointerException, "`dest`");

		if(!isDeviceMemoryPointer(src))
			OPENMPCD_THROW(InvalidArgumentException, "`src`");

		if(!isHostMemoryPointer(dest))
			OPENMPCD_THROW(InvalidArgumentException, "`dest`");
	#endif

	cudaMemcpy(dest, src, sizeof(T) * count, cudaMemcpyDeviceToHost);
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

template<typename T>
	void DeviceMemoryManager::copyElementsFromDeviceToDevice(
		const T* const src, T* const dest, const std::size_t count)
{
	#ifdef OPENMPCD_DEBUG
		if(!src)
			OPENMPCD_THROW(NULLPointerException, "`src`");

		if(!dest)
			OPENMPCD_THROW(NULLPointerException, "`dest`");

		if(!isDeviceMemoryPointer(src))
			OPENMPCD_THROW(InvalidArgumentException, "`src`");

		if(!isDeviceMemoryPointer(dest))
			OPENMPCD_THROW(InvalidArgumentException, "`dest`");
	#endif

	cudaMemcpy(dest, src, sizeof(T) * count, cudaMemcpyDeviceToDevice);
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

template<typename T>
	void DeviceMemoryManager::zeroMemory(
		T* const start, const std::size_t numberOfElements)
{
	#ifdef OPENMPCD_DEBUG
		if(start == NULL)
			OPENMPCD_THROW(NULLPointerException, "`start`");

		if(!isDeviceMemoryPointer(start))
			OPENMPCD_THROW(InvalidArgumentException, "`start`");
	#endif

	cudaMemset(start, 0, sizeof(T) * numberOfElements);
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

template<typename T>
	bool DeviceMemoryManager::elementMemoryEqualOnHostAndDevice(
		const T* const host, const T* const device, const std::size_t count)
{
	#ifdef OPENMPCD_DEBUG
		if(!host)
			OPENMPCD_THROW(NULLPointerException, "`host`");

		if(!device)
			OPENMPCD_THROW(NULLPointerException, "`device`");

		if(!isHostMemoryPointer(host))
			OPENMPCD_THROW(InvalidArgumentException, "`host`");

		if(!isDeviceMemoryPointer(device))
			OPENMPCD_THROW(InvalidArgumentException, "`device`");

		if(count == 0)
			OPENMPCD_THROW(InvalidArgumentException, "`count`");
	#endif

	const std::size_t byteCount = sizeof(T) * count;

	boost::scoped_array<char> copyFromDevice(new char[byteCount]);
	cudaMemcpy(copyFromDevice.get(), device, byteCount, cudaMemcpyDeviceToHost);
	OPENMPCD_CUDA_THROW_ON_ERROR;

	return memcmp(host, copyFromDevice.get(), byteCount) == 0;
}

} //namespace CUDA
} //namespace OpenMPCD

#endif //OPENMPCD_CUDA_IMPLEMENTATIONDETAILS_DEVICEMEMORYMANAGER_HPP
