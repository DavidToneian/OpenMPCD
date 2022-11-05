#include <OpenMPCD/SystemInformation.hpp>

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/runtime.hpp>
#include <OpenMPCD/Exceptions.hpp>

#include <sys/sysinfo.h>
#include <unistd.h>

using namespace OpenMPCD;

const std::string SystemInformation::getHostname()
{
	static const std::size_t bufferSize = HOST_NAME_MAX + 1;

	char buffer[bufferSize];

	if(gethostname(buffer, bufferSize) != 0)
		OPENMPCD_THROW(Exception, "Call to `gethostname` failed.");

	return buffer;
}

std::size_t SystemInformation::getTotalPhysicalMainMemory()
{
	struct sysinfo info;
	if(sysinfo(&info) != 0)
		OPENMPCD_THROW(Exception, "Call to `sysinfo` failed.");

	std::size_t ret = info.mem_unit;
	ret *= info.totalram;

	return ret;
}

const std::vector<SystemInformation::CUDADevice> SystemInformation::getCUDADevices()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	OPENMPCD_CUDA_THROW_ON_ERROR;

	std::vector<CUDADevice> devices;
	devices.reserve(deviceCount);

	for(int i=0; i < deviceCount; ++i)
	{
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, i);
		OPENMPCD_CUDA_THROW_ON_ERROR;

		devices.push_back(CUDADevice());
		devices.back().name = properties.name;
		devices.back().globalMemory = properties.totalGlobalMem;
	}

	return devices;
}
