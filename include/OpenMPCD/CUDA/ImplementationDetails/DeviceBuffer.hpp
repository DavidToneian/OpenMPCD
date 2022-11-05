/**
 * @file
 * Implements functionality of the `OpenMPCD::CUDA::DeviceBuffer` class.
 */

#ifndef OPENMPCD_CUDA_IMPLEMENTATIONDETAILS_DEVICEBUFFER_HPP
#define OPENMPCD_CUDA_IMPLEMENTATIONDETAILS_DEVICEBUFFER_HPP

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

namespace OpenMPCD
{
namespace CUDA
{

template<typename T>
DeviceBuffer<T>::DeviceBuffer(const std::size_t elementCount_)
	: elementCount(elementCount_)
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		elementCount != 0,
		InvalidArgumentException);

	DeviceMemoryManager::allocateMemoryUnregistered(
		&pointer, sizeof(T) * elementCount);
}

template<typename T>
DeviceBuffer<T>::~DeviceBuffer()
{
	DeviceMemoryManager::freeMemoryUnregistered(pointer);
}

template<typename T>
T* DeviceBuffer<T>::getPointer()
{
	return pointer;
}

template<typename T>
const T* DeviceBuffer<T>::getPointer() const
{
	return pointer;
}

template<typename T>
std::size_t DeviceBuffer<T>::getElementCount() const
{
	return elementCount;
}

template<typename T>
void DeviceBuffer<T>::copyFromDevice(const T* const src)
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(src, NULLPointerException);

	DeviceMemoryManager::copyElementsFromDeviceToDevice(
		src, pointer, getElementCount());
}

template<typename T>
void DeviceBuffer<T>::zeroMemory()
{
	DeviceMemoryManager::zeroMemory(getPointer(), getElementCount());
}

} //namespace CUDA
} //namespace OpenMPCD


#endif //OPENMPCD_CUDA_IMPLEMENTATIONDETAILS_DEVICEBUFFER_HPP
