#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/VelocityAutocorrelation/Simple.hpp>

#include <OpenMPCD/CUDA/MPCFluid/Base.hpp>

using namespace OpenMPCD;
using namespace OpenMPCD::CUDA::MPCFluid::Instrumentation::VelocityAutocorrelation;

Simple::Simple(
	const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
	const MPCFluid::Base* const mpcFluid_)
		: Base(sim, devMemMgr, mpcFluid_, mpcFluid_->getParticleCount())
{
}

void Simple::populateCurrentVelocities()
{
	cudaMemcpy(
		currentVelocities, mpcFluid->getDeviceVelocities(),
		3 * sizeof(MPCParticleVelocityType) * numberOfConstituents,
		cudaMemcpyDeviceToDevice);
	OPENMPCD_CUDA_THROW_ON_ERROR;

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
}
