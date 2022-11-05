#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/LogicalEntityMeanSquareDisplacement.hpp>

#include <OpenMPCD/CUDA/MPCFluid/Base.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/FilesystemUtilities.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

#include <fstream>

static const char* const configGroupName =
	"instrumentation.logicalEntityMeanSquareDisplacement";

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
namespace Instrumentation
{

LogicalEntityMeanSquareDisplacement::LogicalEntityMeanSquareDisplacement(
	const Configuration& configuration,
	const CUDA::MPCFluid::Base* const mpcFluid_)
	: mpcFluid(mpcFluid_),
	  sweepsSinceLastMeasurement(0),
	  d_buffer(mpcFluid ? 3 * mpcFluid->getNumberOfLogicalEntities() : 1)
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(mpcFluid_, NULLPointerException);

	const Configuration::Setting configGroup =
		configuration.getSetting(
			"instrumentation.logicalEntityMeanSquareDisplacement");
	if(!isValidConfiguration(configGroup))
		OPENMPCD_THROW(InvalidConfigurationException, "");

	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		mpcFluid->numberOfParticlesPerLogicalEntityIsConstant(),
		InvalidArgumentException);


	configGroup.read("measureEveryNthSweep", &measureEveryNthSweep);
	configGroup.read(
		"measurementArgumentCount", &measurementArgumentCount);

	OPENMPCD_DEBUG_ASSERT(measureEveryNthSweep > 0);
	OPENMPCD_DEBUG_ASSERT(measurementArgumentCount > 0);

	const std::size_t bufferElementCount =
		3 * mpcFluid->getNumberOfLogicalEntities();
	for(std::size_t i = 0; i < measurementArgumentCount + 1; ++i)
		snapshots.push_back(new MPCParticlePositionType[bufferElementCount]);
}

LogicalEntityMeanSquareDisplacement::~LogicalEntityMeanSquareDisplacement()
{
	for(std::size_t i = 0; i < snapshots.size(); ++i)
		delete[] snapshots[i];
}

bool LogicalEntityMeanSquareDisplacement::isConfigured(
	const Configuration& config)
{
	return config.has(configGroupName);
}

bool LogicalEntityMeanSquareDisplacement::isValidConfiguration(
	const Configuration::Setting& group)
{
	if(!group.has("measureEveryNthSweep"))
		return false;

	if(!group.has("measurementArgumentCount"))
		return false;

	try
	{
		unsigned int measureEveryNthSweep;
		group.read("measureEveryNthSweep", &measureEveryNthSweep);

		if(measureEveryNthSweep == 0)
			return false;


		unsigned int measurementArgumentCount;
		group.read(
			"measurementArgumentCount", &measurementArgumentCount);

		if(measurementArgumentCount == 0)
			return false;
	}
	catch(const InvalidConfigurationException&)
	{
		return false;
	}

	return true;
}

void LogicalEntityMeanSquareDisplacement::measure()
{
	++sweepsSinceLastMeasurement;
	if(sweepsSinceLastMeasurement != measureEveryNthSweep)
	{
		if(meanSquareDisplacements.size() != 0)
			return;
	}

	sweepsSinceLastMeasurement = 0;


	OPENMPCD_DEBUG_ASSERT(measurementArgumentCount > 0);

	std::vector<MPCParticlePositionType*> buffers;
	//buffers[0] holds the snapshot for the current
	//measurement time, buffers[1] holds the snapshot for the previous one, etc.
	buffers.reserve(measurementArgumentCount + 1);
	for(unsigned int i = 0; i < measurementArgumentCount + 1; ++i)
		buffers.push_back(NULL);

	const unsigned int currentMeasurementTime = meanSquareDisplacements.size();

	for(unsigned int i = 0; i < buffers.size(); ++i)
	{
		if(i > currentMeasurementTime)
			continue;

		buffers[i] =
			snapshots[(currentMeasurementTime - i) % buffers.size()];
	}

	mpcFluid->saveLogicalEntityCentersOfMassToDeviceMemory(
		d_buffer.getPointer());
	DeviceMemoryManager::copyElementsFromDeviceToHost(
		d_buffer.getPointer(), buffers[0],
		3 * mpcFluid->getNumberOfLogicalEntities());


	meanSquareDisplacements.push_back(std::vector<MPCParticlePositionType>());

	for(unsigned int i = 1; i < buffers.size(); ++i)
	{
		if(buffers[i] == NULL)
			break;

		MPCParticlePositionType result = 0;
		for(unsigned int e = 0; e < mpcFluid->getNumberOfLogicalEntities(); ++e)
		{
			const RemotelyStoredVector<const MPCParticlePositionType>
				R0(buffers[0], e);
			const RemotelyStoredVector<const MPCParticlePositionType>
				RT(buffers[i], e);

			const Vector3D<MPCParticlePositionType> distance = R0 - RT;

			result += distance.getMagnitudeSquared();
		}
		result /= mpcFluid->getNumberOfLogicalEntities();

		meanSquareDisplacements[currentMeasurementTime].push_back(result);
	}
}

unsigned int LogicalEntityMeanSquareDisplacement::getMaximumMeasurementTime() const
{
	OPENMPCD_DEBUG_ASSERT(measurementArgumentCount > 0);

	return measurementArgumentCount;
}

unsigned int LogicalEntityMeanSquareDisplacement::getMeasurementCount() const
{
	return meanSquareDisplacements.size();
}

MPCParticlePositionType
LogicalEntityMeanSquareDisplacement::getMeanSquareDisplacement(
	const unsigned int t,
	const unsigned int T) const
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		t < getMeasurementCount(), InvalidArgumentException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		T < getMeasurementCount(), InvalidArgumentException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		t < T, InvalidArgumentException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		T - t <= getMaximumMeasurementTime(), InvalidArgumentException);

	OPENMPCD_DEBUG_ASSERT(T < meanSquareDisplacements.size());
	OPENMPCD_DEBUG_ASSERT(T - t - 1 < meanSquareDisplacements[T].size());
	return meanSquareDisplacements[T][T - t - 1];
}

void LogicalEntityMeanSquareDisplacement::save(std::ostream& stream)
{
	const std::streamsize oldPrecision =
		stream.precision(std::numeric_limits<FP>::digits10 + 2);

	for(unsigned int t = 0; t < getMeasurementCount(); ++t)
	{
		for(unsigned int dT = 1; dT <= measurementArgumentCount; ++dT)
		{
			const unsigned int T = t + dT;
			if(T >= getMeasurementCount())
				break;

			stream << t << "\t" << dT;
			stream << "\t" << getMeanSquareDisplacement(t, T);
			stream << "\n";
		}
	}

	stream.precision(oldPrecision);
}

void LogicalEntityMeanSquareDisplacement::save(const std::string& rundir)
{
	const std::string path =
		rundir + "/logicalEntityMeanSquareDisplacement.data";

	FilesystemUtilities::ensureParentDirectory(path);

	std::ofstream file(path.c_str(), std::ios::trunc);

	save(file);
}

} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD
