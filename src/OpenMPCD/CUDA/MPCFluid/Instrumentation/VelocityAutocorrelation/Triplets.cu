#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/VelocityAutocorrelation/Triplets.hpp>

#include <OpenMPCD/CUDA/MPCFluid/Base.hpp>
#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Triplets.hpp>

using namespace OpenMPCD;
using namespace OpenMPCD::CUDA::MPCFluid::Instrumentation::VelocityAutocorrelation;

Triplets::Triplets(
	const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
	const MPCFluid::Base* const mpcFluid_)
		: Base(sim, devMemMgr, mpcFluid_, mpcFluid_->getParticleCount() / 3)
{
}

void Triplets::populateCurrentVelocities()
{
	DeviceCode::getCenterOfMassVelocities_triplet(
		mpcFluid->getParticleCount(),
		mpcFluid->getDeviceVelocities(),
		currentVelocities);
}
