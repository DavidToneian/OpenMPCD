#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/VelocityAutocorrelation/Chains.hpp>

#include <OpenMPCD/CUDA/MPCFluid/Base.hpp>
#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Chains.hpp>

using namespace OpenMPCD;
using namespace OpenMPCD::CUDA::MPCFluid::Instrumentation::VelocityAutocorrelation;

Chains::Chains(
	const unsigned int chainLength_,
	const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
	const MPCFluid::Base* const mpcFluid_)
		: Base(
			sim,
			devMemMgr,
			mpcFluid_,
			mpcFluid_ == NULL ? 1 :
				mpcFluid_->getParticleCount() / chainLength_
		  ),
		  chainLength(chainLength_)
{
}

void Chains::populateCurrentVelocities()
{
	DeviceCode::getCenterOfMassVelocities_chain(
		mpcFluid->getParticleCount(),
		chainLength,
		mpcFluid->getDeviceVelocities(),
		currentVelocities);
}
