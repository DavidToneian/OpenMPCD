#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/VelocityAutocorrelation/Doublets.hpp>

#include <OpenMPCD/CUDA/MPCFluid/Base.hpp>
#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Doublets.hpp>

using namespace OpenMPCD;
using namespace OpenMPCD::CUDA::MPCFluid::Instrumentation::VelocityAutocorrelation;

Doublets::Doublets(
	const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
	const MPCFluid::Base* const mpcFluid_)
		: Base(sim, devMemMgr, mpcFluid_, mpcFluid_->getParticleCount() / 2)
{
}

void Doublets::populateCurrentVelocities()
{
	DeviceCode::getCenterOfMassVelocities_doublet(
		mpcFluid->getParticleCount(),
		mpcFluid->getDeviceVelocities(),
		currentVelocities);
}
