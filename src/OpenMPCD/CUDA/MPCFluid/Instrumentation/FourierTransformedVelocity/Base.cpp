#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/FourierTransformedVelocity/Base.hpp>

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/runtime.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>
#include <OpenMPCD/FilesystemUtilities.hpp>

#include <fstream>
#include <sstream>

using namespace OpenMPCD;
using namespace OpenMPCD::CUDA::MPCFluid::Instrumentation::FourierTransformedVelocity;

Base::Base(const Simulation* const sim, DeviceMemoryManager* devMemMgr, const MPCFluid::Base* const mpcFluid_,
           const unsigned int numberOfConstituents)
	: simulation(sim), deviceMemoryManager(devMemMgr), mpcFluid(mpcFluid_),
	  realBuffer(NULL), imagBuffer(NULL)
{
	readConfig();

	deviceMemoryManager->allocateMemory(&realBuffer, 3 * numberOfConstituents);
	deviceMemoryManager->allocateMemory(&imagBuffer, 3 * numberOfConstituents);
}

Base::~Base()
{
	deviceMemoryManager->freeMemory(realBuffer);
	deviceMemoryManager->freeMemory(imagBuffer);
}

void Base::save(const std::string& rundir) const
{
	saveFourierTransformedVelocities(rundir+"/fourierTransformedVelocities");
}

bool Base::isConfigured(const Simulation* const sim)
{
	if(!sim)
		OPENMPCD_THROW(NULLPointerException, "`sim`");

	return sim->getConfiguration().has("instrumentation.fourierTransformedVelocity");
}

void Base::readConfig()
{
	const Configuration& config = simulation->getConfiguration();

	const Configuration::List k_n_config = config.getList("instrumentation.fourierTransformedVelocity.k_n");

	for(unsigned int i=0; i<k_n_config.getSize(); ++i)
	{
		const Configuration::List current = k_n_config.getList(i);

		std::vector<MPCParticlePositionType> n;

		n.push_back(current.read<MPCParticlePositionType>(0));
		n.push_back(current.read<MPCParticlePositionType>(1));
		n.push_back(current.read<MPCParticlePositionType>(2));

		k_n.push_back(n);
	}

	fourierTransformedVelocities.resize(k_n.size());
}

void Base::saveFourierTransformedVelocities(const std::string& datadir) const
{
	FilesystemUtilities::ensureDirectory(datadir);

	for(unsigned int i=0; i<k_n.size(); ++i)
	{
		std::stringstream path;
		path<<datadir<<"/"<<i<<".data";

		std::ofstream file(path.str().c_str(), std::ios::trunc);
		file.precision(std::numeric_limits<FP>::digits10 + 2);

		saveFourierTransformedVelocities(i, file);
	}
}

void Base::saveFourierTransformedVelocities(const unsigned int index, std::ostream& stream) const
{
	if(index >= k_n.size())
		OPENMPCD_THROW(OutOfBoundsException, "index");

	const Configuration& config          = simulation->getConfiguration();
	const Configuration::List k_n_config = config.getList("instrumentation.fourierTransformedVelocity.k_n");
	const Configuration::List current    = k_n_config.getList(index);

	stream<<"#k_n:\t"<<current.read<FP>(0)<<"\t"<<current.read<FP>(1)<<"\t"<<current.read<FP>(2)<<"\n";

	for(unsigned int i=0; i<fourierTransformedVelocities[index].size(); ++i)
	{
		const FP mpcTime = fourierTransformedVelocities[index][i].first;
		const Vector3D<std::complex<MPCParticleVelocityType> >& value
			= fourierTransformedVelocities[index][i].second;

		stream
			<<mpcTime<<"\t"
			<<value.getX().real()<<"\t"<<value.getX().imag()<<"\t"
			<<value.getY().real()<<"\t"<<value.getY().imag()<<"\t"
			<<value.getZ().real()<<"\t"<<value.getZ().imag()<<"\n";
	}
}
