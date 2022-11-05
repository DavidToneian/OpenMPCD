/**
 * @file
 * Main file for the testing suite.
 */

#define CATCH_CONFIG_RUNNER
#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Device.hpp>
#include <OpenMPCD/MPI.hpp>

int main(int argc, char* argv[])
{
	OpenMPCD::MPI mpi;

	OpenMPCD::CUDA::Device device;
	if(device.getStackSizePerThread() < 2048)
		device.setStackSizePerThread(2048);

	Catch::Session session;

	{
		const int ret = session.applyCommandLine(argc, const_cast<const char**>(argv));
		if(ret != 0)
			return ret;
	}

	return session.run();
}
