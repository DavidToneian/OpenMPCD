#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/GaussianChains.hpp>

#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/FourierTransformedVelocity/Chains.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/VelocityAutocorrelation/Chains.hpp>

using namespace OpenMPCD::CUDA::MPCFluid::Instrumentation;

GaussianChains::GaussianChains(
	const unsigned int chainLength,
	const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
	const MPCFluid::GaussianChains* const mpcFluid_)
		: Base(sim->getConfiguration(), mpcFluid_),
		  simulation(sim), mpcFluid(mpcFluid_),
		  squaredBondLengths(NULL)
{
	if(FourierTransformedVelocity::Chains::isConfigured(simulation))
	{
		fourierTransformedVelocity =
			new
			FourierTransformedVelocity::Chains(
				chainLength, sim, devMemMgr, mpcFluid_);
	}

	if(VelocityAutocorrelation::Chains::isConfigured(simulation))
	{
		velocityAutocorrelation =
			new
			VelocityAutocorrelation::Chains(
				chainLength, sim, devMemMgr, mpcFluid_);
	}

	const Configuration& config = sim->getConfiguration();
	if(config.has("instrumentation.gaussianChains.squaredBondLengths"))
	{
		if(chainLength < 2)
		{
			OPENMPCD_THROW(
				Exception,
				"Cannot measure bond lengths with less than two beads.");
		}

		squaredBondLengths =
			new std::vector<OnTheFlyStatisticsDDDA<MPCParticlePositionType> >();
		squaredBondLengths->resize(chainLength - 1);
	}
}

void GaussianChains::measureSpecific()
{
	if(squaredBondLengths)
	{
		mpcFluid->fetchFromDevice();
		const unsigned int chainCount = mpcFluid->getNumberOfLogicalEntities();
		const unsigned int particlesPerChain =
			mpcFluid->getNumberOfParticlesPerLogicalEntity();

		OPENMPCD_DEBUG_ASSERT(particlesPerChain >= 2);

		for(unsigned int chain = 0; chain < chainCount; ++chain)
		{
			const unsigned int firstParticleID = chain * particlesPerChain;

			for(unsigned int p = 0; p < particlesPerChain - 1; ++p)
			{
				const Vector3D<MPCParticlePositionType> distance =
					mpcFluid->getPosition(firstParticleID + p)
					-
					mpcFluid->getPosition(firstParticleID + p + 1);

				(*squaredBondLengths)[p].addDatum(
					distance.getMagnitudeSquared());
			}
		}
	}
}

void GaussianChains::saveSpecific(const std::string& rundir) const
{
	if(squaredBondLengths)
	{
		const std::string filePath =
			rundir + "/gaussianChains--squaredBondLengths.txt";
		std::ofstream file(filePath.c_str(), std::ios::out);

		file << "#bond-number\t" << "mean-square-bond-length\t"
		     << "sample-size\t" << "sample-standard-deviation\t"
		     << "DDDA-optimal-block-ID\t"
		     << "DDDA-optimal-standard-error-of-the-mean\t"
		     << "DDDA-optimal-standard-error-of-the-mean-is-reliable\n";

		for(std::size_t bond = 0; bond < squaredBondLengths->size(); ++bond)
		{
			const MPCParticlePositionType standardDeviation =
				sqrt((*squaredBondLengths)[bond].getBlockVariance(0));

			file << bond << "\t";
			file << (*squaredBondLengths)[bond].getSampleMean() << "\t";
			file << (*squaredBondLengths)[bond].getSampleSize() << "\t";
			file << standardDeviation << "\t";
			file <<
				(*squaredBondLengths)[bond].getOptimalBlockIDForStandardErrorOfTheMean()
				<< "\t";
			file <<
				(*squaredBondLengths)[bond].getOptimalStandardErrorOfTheMean()
				<< "\t";
			file <<
				((*squaredBondLengths)[bond].optimalStandardErrorOfTheMeanEstimateIsReliable()
				? 1 : 0) << "\n";
		}
	}
}
