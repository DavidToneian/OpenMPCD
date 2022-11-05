#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/GaussianRods.hpp>

#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/FourierTransformedVelocity/Doublets.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/VelocityAutocorrelation/Doublets.hpp>

using namespace OpenMPCD::CUDA::MPCFluid::Instrumentation;

GaussianRods::GaussianRods(
	const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
	const MPCFluid::GaussianRods* const mpcFluid_)
		: Base(sim->getConfiguration(), mpcFluid_),
		  simulation(sim), mpcFluid(mpcFluid_),
		  bondLengthHistogram("gaussianRods.bondLengthHistogram", sim->getConfiguration())
{
	fourierTransformedVelocity = new FourierTransformedVelocity::Doublets(sim, devMemMgr, mpcFluid_);
	velocityAutocorrelation    = new VelocityAutocorrelation::Doublets(sim, devMemMgr, mpcFluid_);
}

void GaussianRods::measureSpecific()
{
	static const Vector3D<MPCParticleVelocityType> flowDirection(1, 0, 0);

	for(unsigned int i=0; i<mpcFluid->getParticleCount(); i+=2)
	{
		const RemotelyStoredVector<const MPCParticlePositionType> r_1 = mpcFluid->getPosition(i);
		const RemotelyStoredVector<const MPCParticlePositionType> r_2 = mpcFluid->getPosition(i+1);

		const Vector3D<MPCParticlePositionType> R = r_2 - r_1;

		bondLengthHistogram.fill(R.magnitude());
	}
}

void GaussianRods::saveSpecific(const std::string& rundir) const
{
	bondLengthHistogram.save(rundir+"/gaussianRods/bondLengthHistogram.data");
}
