#include <OpenMPCD/CUDA/Device.hpp>

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

#include <iomanip>
#include <sstream>

namespace OpenMPCD
{
namespace CUDA
{

Device::Device()
{
	cudaGetDevice(&deviceID);
	OPENMPCD_CUDA_THROW_ON_ERROR;

	cudaGetDeviceProperties(&properties, deviceID);
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

unsigned int Device::getDeviceCount()
{
	int ret = 0;
	cudaGetDeviceCount(&ret);
	OPENMPCD_CUDA_THROW_ON_ERROR;

	OPENMPCD_DEBUG_ASSERT(ret >= 1);

	return static_cast<unsigned int>(ret);
}

unsigned int Device::getPCIDomainID() const
{
	OPENMPCD_DEBUG_ASSERT(properties.pciDomainID >= 0);

	return static_cast<unsigned int>(properties.pciDomainID);
}

unsigned int Device::getPCIBusID() const
{
	OPENMPCD_DEBUG_ASSERT(properties.pciBusID >= 0);

	return static_cast<unsigned int>(properties.pciBusID);
}

unsigned int Device::getPCISlotID() const
{
	OPENMPCD_DEBUG_ASSERT(properties.pciDeviceID >= 0);

	return static_cast<unsigned int>(properties.pciDeviceID);
}

const std::string Device::getPCIAddressString() const
{
	std::stringstream ss;
	ss << std::hex << std::setfill('0');
	ss << std::setw(4) <<getPCIDomainID() << ":";
	ss << std::setw(2) <<getPCIBusID() << ":";
	ss << std::setw(2) <<getPCISlotID();

	return ss.str();
}

std::size_t Device::getStackSizePerThread() const
{
	std::size_t ret;
	if(cudaDeviceGetLimit(&ret, cudaLimitStackSize) != cudaSuccess)
		OPENMPCD_THROW(Exception, "Failed to get stack size limit.");
	return ret;
}

void Device::setStackSizePerThread(const std::size_t value) const
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(value != 0, InvalidArgumentException);

	if(cudaDeviceSetLimit(cudaLimitStackSize, value) != cudaSuccess)
		OPENMPCD_THROW(Exception, "Failed to set stack size limit.");
}


} //namespace CUDA
} //namespace OpenMPCD
