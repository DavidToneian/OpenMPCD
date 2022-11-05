#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/NormalModeAutocorrelation.hpp>

#include <OpenMPCD/CUDA/MPCFluid/Base.hpp>
#include <OpenMPCD/CUDA/NormalMode.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/FilesystemUtilities.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

#include <fstream>

static const char* const configGroupName =
	"instrumentation.normalModeAutocorrelation";

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
namespace Instrumentation
{

NormalModeAutocorrelation::NormalModeAutocorrelation(
	const Configuration& configuration,
	const CUDA::MPCFluid::Base* const mpcFluid_)
	: mpcFluid(mpcFluid_),
	  sweepsSinceLastMeasurement(0),
	  shift(0)
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(mpcFluid_, NULLPointerException);

	const Configuration::Setting configGroup =
		configuration.getSetting("instrumentation.normalModeAutocorrelation");
	if(!isValidConfiguration(configGroup))
		OPENMPCD_THROW(InvalidConfigurationException, "");

	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		mpcFluid->numberOfParticlesPerLogicalEntityIsConstant(),
		InvalidArgumentException);


	configGroup.read("measureEveryNthSweep", &measureEveryNthSweep);
	configGroup.read(
		"autocorrelationArgumentCount", &autocorrelationArgumentCount);

	if(configGroup.has("shift"))
		configGroup.read("shift", &shift);

	OPENMPCD_DEBUG_ASSERT(measureEveryNthSweep > 0);
	OPENMPCD_DEBUG_ASSERT(autocorrelationArgumentCount > 0);

	const std::size_t bufferElementCount =
		mpcFluid->getNumberOfLogicalEntities() *
		(mpcFluid->getNumberOfParticlesPerLogicalEntity() + 1);
	for(std::size_t i = 0; i < autocorrelationArgumentCount; ++i)
	{
		snapshots.push_back(
			new DeviceBuffer<MPCParticlePositionType>(bufferElementCount));
	}
}

NormalModeAutocorrelation::~NormalModeAutocorrelation()
{
	for(std::size_t i = 0; i < snapshots.size(); ++i)
		delete snapshots[i];
}

bool NormalModeAutocorrelation::isConfigured(const Configuration& config)
{
	return config.has(configGroupName);
}

bool NormalModeAutocorrelation::isValidConfiguration(
	const Configuration::Setting& group)
{
	if(!group.has("measureEveryNthSweep"))
		return false;

	if(!group.has("autocorrelationArgumentCount"))
		return false;

	try
	{
		unsigned int measureEveryNthSweep;
		group.read("measureEveryNthSweep", &measureEveryNthSweep);

		if(measureEveryNthSweep == 0)
			return false;


		unsigned int autocorrelationArgumentCount;
		group.read(
			"autocorrelationArgumentCount", &autocorrelationArgumentCount);

		if(autocorrelationArgumentCount == 0)
			return false;
	}
	catch(const InvalidConfigurationException&)
	{
		return false;
	}

	return true;
}

void NormalModeAutocorrelation::measure()
{
	++sweepsSinceLastMeasurement;
	if(sweepsSinceLastMeasurement != measureEveryNthSweep)
	{
		if(autocorrelations.size() != 0)
			return;
	}

	sweepsSinceLastMeasurement = 0;

	std::vector<DeviceBuffer<MPCParticlePositionType>* > buffers;
	//buffers[0] holds the normal coordinate snapshot for the current
	//measurement time, buffers[1] holds the snapshot for the previous one, etc.
	buffers.reserve(autocorrelationArgumentCount);
	for(unsigned int i = 0; i < autocorrelationArgumentCount; ++i)
		buffers.push_back(NULL);

	const unsigned int currentMeasurementTime = autocorrelations.size();

	OPENMPCD_DEBUG_ASSERT(autocorrelationArgumentCount > 0);
	for(unsigned int i = 0; i < autocorrelationArgumentCount; ++i)
	{
		if(i > currentMeasurementTime)
			continue;

		buffers[i] =
			snapshots[
				(currentMeasurementTime - i) % autocorrelationArgumentCount];
	}


	NormalMode::computeNormalCoordinates(
		mpcFluid->getNumberOfParticlesPerLogicalEntity(),
		mpcFluid->getNumberOfLogicalEntities(),
		mpcFluid->getDevicePositions(),
		buffers[0]->getPointer(),
		shift);


	autocorrelations.push_back(
		std::vector<std::vector<MPCParticlePositionType> >());


	for(unsigned int i = 0; i < buffers.size(); ++i)
	{
		if(buffers[i] == NULL)
			break;

		const std::vector<MPCParticlePositionType> result =
			NormalMode::getAverageNormalCoordinateAutocorrelation(
				mpcFluid->getNumberOfParticlesPerLogicalEntity(),
				mpcFluid->getNumberOfLogicalEntities(),
				buffers[i]->getPointer(),
				buffers[0]->getPointer());

		autocorrelations[currentMeasurementTime].push_back(result);
	}
}

unsigned int NormalModeAutocorrelation::getMeasurementCount() const
{
	return autocorrelations.size();
}

unsigned int NormalModeAutocorrelation::getMaximumCorrelationTime() const
{
	OPENMPCD_DEBUG_ASSERT(autocorrelationArgumentCount > 0);

	return autocorrelationArgumentCount - 1;
}

MPCParticlePositionType NormalModeAutocorrelation::getAutocorrelation(
	const unsigned int t,
	const unsigned int T,
	const unsigned int normalMode) const
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		t < getMeasurementCount(), InvalidArgumentException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		T < getMeasurementCount(), InvalidArgumentException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		t <= T, InvalidArgumentException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		T - t <= getMaximumCorrelationTime(), InvalidArgumentException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		normalMode <= mpcFluid->getNumberOfParticlesPerLogicalEntity(),
		InvalidArgumentException);

	OPENMPCD_DEBUG_ASSERT(T < autocorrelations.size());
	OPENMPCD_DEBUG_ASSERT(T - t < autocorrelations[T].size());
	OPENMPCD_DEBUG_ASSERT(normalMode < autocorrelations[T][T - t].size());
	return autocorrelations[T][T - t][normalMode];
}

void NormalModeAutocorrelation::save(std::ostream& stream)
{
	const std::streamsize oldPrecision =
		stream.precision(std::numeric_limits<FP>::digits10 + 2);

	const unsigned int normalModeCount =
		mpcFluid->getNumberOfParticlesPerLogicalEntity() + 1;
	for(unsigned int t = 0; t < getMeasurementCount(); ++t)
	{
		for(unsigned int dT = 0; dT < autocorrelationArgumentCount; ++dT)
		{
			const unsigned int T = t + dT;
			if(T >= getMeasurementCount())
				break;

			stream << t << "\t" << dT;
			for(unsigned int n = 0; n < normalModeCount; ++n)
				stream << "\t" << getAutocorrelation(t, T, n);
			stream << "\n";
		}
	}

	stream.precision(oldPrecision);
}

void NormalModeAutocorrelation::save(const std::string& rundir)
{
	const std::string path = rundir + "/normalModeAutocorrelations.data";

	FilesystemUtilities::ensureParentDirectory(path);

	std::ofstream file(path.c_str(), std::ios::trunc);

	save(file);
}

} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD
