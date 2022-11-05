#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/FourierTransformedVelocity/Triplets.hpp>

#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/FourierTransformedVelocity/DeviceCode/fourierTransformedVelocity.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>

#include <boost/math/constants/constants.hpp>

using namespace OpenMPCD;
using namespace OpenMPCD::CUDA::MPCFluid::Instrumentation::FourierTransformedVelocity;

Triplets::Triplets(
	const Simulation* const sim, DeviceMemoryManager* devMemMgr,
	const MPCFluid::Base* const mpcFluid_)
		: Base(sim, devMemMgr, mpcFluid_, mpcFluid_->getParticleCount() / 3)
{
}

void Triplets::measure()
{
	static const FP pi = boost::math::constants::pi<FP>();

	const FP factorX = 2*pi / simulation->getSimulationBoxSizeX();
	const FP factorY = 2*pi / simulation->getSimulationBoxSizeY();
	const FP factorZ = 2*pi / simulation->getSimulationBoxSizeZ();

	const FP mpcTime = simulation->getMPCTime();

	for(unsigned int i=0; i<k_n.size(); ++i)
	{
		#ifdef OPENMPCD_DEBUG
			if(k_n[i].size()!=3)
				OPENMPCD_THROW(Exception, "Unexpected");
		#endif

		const Vector3D<MPCParticlePositionType> k(
				k_n[i][0] * factorX,
				k_n[i][1] * factorY,
				k_n[i][2] * factorZ);

		const Vector3D<std::complex<MPCParticleVelocityType> > result =
			DeviceCode::calculateVelocityInFourierSpace_tripletMPCFluid(
				mpcFluid->getParticleCount() / 3,
				mpcFluid->getDevicePositions(),
				mpcFluid->getDeviceVelocities(),
				k,
				realBuffer,
				imagBuffer
				);

		fourierTransformedVelocities[i].push_back(std::make_pair(mpcTime, result));
	}
}
