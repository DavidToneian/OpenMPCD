#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/Simple.hpp>

#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/FourierTransformedVelocity/Simple.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/VelocityAutocorrelation/Simple.hpp>

using namespace OpenMPCD;
using namespace OpenMPCD::CUDA::MPCFluid::Instrumentation;

Simple::Simple(
	const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
	const MPCFluid::Simple* const mpcFluid_)
	: Base(sim->getConfiguration(), mpcFluid_)
{
	if(FourierTransformedVelocity::Simple::isConfigured(sim))
		fourierTransformedVelocity = new FourierTransformedVelocity::Simple(sim, devMemMgr, mpcFluid_);

	if(VelocityAutocorrelation::Simple::isConfigured(sim))
		velocityAutocorrelation = new VelocityAutocorrelation::Simple(sim, devMemMgr, mpcFluid_);
}

Simple::~Simple()
{
}

void Simple::measureSpecific()
{
}

void Simple::saveSpecific(const std::string& rundir) const
{
}
