#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/runtime.hpp>
#include <OpenMPCD/Exceptions.hpp>

#include <iostream>
#include <sstream>

using namespace OpenMPCD::CUDA;

DeviceMemoryManager::DeviceMemoryManager()
	: autofree(false)
{
}

DeviceMemoryManager::~DeviceMemoryManager()
{
	if(allocatedBuffers.empty())
		return;

	if(autofree)
	{
		for(std::set<const void*>::const_iterator it=allocatedBuffers.begin(); it!=allocatedBuffers.end(); ++it)
			cudaFree(const_cast<void*>(*it));

		return;
	}

	std::stringstream message;

	message<<"Unfreed Device memory:";

	for(std::set<const void*>::const_iterator it=allocatedBuffers.begin(); it!=allocatedBuffers.end(); ++it)
		message<<" "<<(*it);

	std::cerr<<message.str()<<"\n";
}

void DeviceMemoryManager::freeMemory(void* const pointer)
{
	if(pointer == NULL)
		return;

	if(allocatedBuffers.count(pointer)==0)
		OPENMPCD_THROW(MemoryManagementException, "Tried to free invalid pointer.");

	allocatedBuffers.erase(pointer);

	freeMemoryUnregistered(pointer);
}

void DeviceMemoryManager::freeMemoryUnregistered(void* const pointer)
{
	if(pointer == NULL)
		return;

	#ifdef OPENMPCD_DEBUG
		if(!isDeviceMemoryPointer(pointer))
			OPENMPCD_THROW(MemoryManagementException, "not a Device pointer");

		if(cudaPeekAtLastError() != cudaSuccess)
		{
			std::stringstream message;
			message <<
				"Failure detected in "
				"CUDA::DeviceMemoryManager::freeMemoryUnregistered before the "
				"call to cudaFree:\n";
			message << cudaGetErrorString(cudaPeekAtLastError()) << "\n";

			std::cerr << message.str();

			OPENMPCD_THROW(Exception, message.str());
		}
	#endif

	cudaFree(pointer);
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

bool DeviceMemoryManager::isDeviceMemoryPointer(const void* const ptr)
{
	cudaPointerAttributes attributes;
	cudaPointerGetAttributes(&attributes, ptr);

	switch(cudaGetLastError())
	{
		case cudaSuccess:
			break;

		case cudaErrorInvalidDevice:
			OPENMPCD_THROW(Exception, "");
			break;

		case cudaErrorInvalidValue:
			return false;

		default:
			OPENMPCD_THROW(Exception, "");
	}

	return attributes.memoryType == cudaMemoryTypeDevice;
}

bool DeviceMemoryManager::isHostMemoryPointer(const void* const ptr)
{
	if(!ptr)
		return false;

	return !isDeviceMemoryPointer(ptr);
}

void* DeviceMemoryManager::allocateMemoryInternal(const std::size_t bufferSize)
{
	if(bufferSize==0)
		return NULL;

	void* const pointer = allocateMemoryInternalUnregistered(bufferSize);

	#ifdef OPENMPCD_DEBUG
		if(allocatedBuffers.count(pointer) != 0)
		{
			OPENMPCD_THROW(
				MemoryManagementException,
				"cudaMalloc succeeded, but yielded a used address.");
		}
	#endif

	allocatedBuffers.insert(pointer);

	return pointer;
}

void* DeviceMemoryManager::allocateMemoryInternalUnregistered(
	const std::size_t bufferSize)
{
	if(bufferSize==0)
		return NULL;

	void* pointer=NULL;
	const cudaError_t result = cudaMalloc(&pointer, bufferSize);

	if(result!=cudaSuccess)
	{
		std::stringstream message;
		message<<"cudaMalloc failed when trying to allocate ";
		message<<bufferSize<<" bytes. Error code: "<<result;
		message<<". Error message: "<<cudaGetErrorString(result);

		cudaGetLastError(); //reset last error to cudaSuccess

		OPENMPCD_THROW(MemoryManagementException, message.str());
	}

	#ifdef OPENMPCD_DEBUG
		if(pointer == NULL)
		{
			OPENMPCD_THROW(
				MemoryManagementException,
				"cudaMalloc succeeded, but yielded NULL pointer.");
		}
	#endif

	return pointer;
}
