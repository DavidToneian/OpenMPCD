#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/VelocityAutocorrelation/Base.hpp>

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/runtime.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>
#include <OpenMPCD/FilesystemUtilities.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>

#include <fstream>

using namespace OpenMPCD;
using namespace OpenMPCD::CUDA::MPCFluid::Instrumentation::VelocityAutocorrelation;

#ifdef OPENMPCD_DEBUG_CUDA_MPCFLUID_INSTRUMENTATION_VELOCITYAUTOCORRELATION
	#include <sstream>
#endif

Base::Snapshot::Snapshot(const unsigned int numberOfConstituents, DeviceMemoryManager* const devMemMgr)
	: snapshotTime(-1), buffer(NULL), deviceMemoryManager(devMemMgr)
{
	deviceMemoryManager->allocateMemory(&buffer, 3 * numberOfConstituents);
}

Base::Snapshot::~Snapshot()
{
	deviceMemoryManager->freeMemory(buffer);
}

Base::Base(
	const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
	const MPCFluid::Base* const mpcFluid_,
	const unsigned int numberOfConstituents_)
		: simulation(sim), deviceMemoryManager(devMemMgr),
		  mpcFluid(mpcFluid_), numberOfConstituents(numberOfConstituents_)
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(sim != NULL, NULLPointerException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(devMemMgr != NULL, NULLPointerException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(mpcFluid_ != NULL, NULLPointerException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		numberOfConstituents_ != 0,
		InvalidArgumentException);

	readConfig();

	deviceMemoryManager->allocateMemory(&currentVelocities,  3 * numberOfConstituents);
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

Base::~Base()
{
	deviceMemoryManager->freeMemory(currentVelocities);

	for(unsigned int i=0; i<snapshots.size(); ++i)
		delete snapshots[i];
}

void Base::measure()
{
	populateCurrentVelocities();

	updateSnapshots();

	for(unsigned int i=0; i<snapshots.size(); ++i)
	{
		if(snapshots[i]->getSnapshotTime() >= 0)
			measureWithSnapshot(snapshots[i]);
	}
}

void Base::save(const std::string& rundir) const
{
	saveAutocorrelations(rundir+"/velocityAutocorrelations.data");
}

bool Base::isConfigured(const Simulation* const sim)
{
	if(!sim)
		OPENMPCD_THROW(NULLPointerException, "`sim`");

	return sim->getConfiguration().has("instrumentation.velocityAutocorrelation");
}

void Base::readConfig()
{
	const Configuration& config = simulation->getConfiguration();

	config.read("instrumentation.velocityAutocorrelation.measurementTime", &measurementTime);


	const unsigned int snapshotCount =
		config.read<unsigned int>("instrumentation.velocityAutocorrelation.snapshotCount");

	if(snapshotCount==0)
		OPENMPCD_THROW(
			InvalidConfigurationException,
			"instrumentation.velocityAutocorrelation.snapshotCount must be greater than 0.");

	if(snapshotCount * simulation->getMPCTimestep() > measurementTime - 1e-6)
		OPENMPCD_THROW(
			InvalidConfigurationException,
			"instrumentation.velocityAutocorrelation.snapshotCount is too large compared to "
			"instrumentation.velocityAutocorrelation.measurementTime");

	snapshots.reserve(snapshotCount);

	for(unsigned int i=0; i<snapshotCount; ++i)
		snapshots.push_back(new Snapshot(numberOfConstituents, deviceMemoryManager));
}

void Base::updateSnapshots()
{
	const FP mpcTime = simulation->getMPCTime();

	for(unsigned int i=0; i<snapshots.size(); ++i)
	{
		Snapshot* const snapshot = snapshots[i];

		if(!initializeSnapshotIfAppropriate(i))
			return;

		if(mpcTime - snapshot->getSnapshotTime() - 1e-6 > measurementTime)
			updateSnapshot(snapshot);
	}
}

void Base::updateSnapshot(Snapshot* const snapshot)
{
	cudaMemcpy(
		snapshot->getBuffer(), currentVelocities,
		3 * sizeof(MPCParticleVelocityType) * numberOfConstituents,
		cudaMemcpyDeviceToDevice);
	OPENMPCD_CUDA_THROW_ON_ERROR;

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	snapshot->setSnapshotTime(simulation->getMPCTime());
}

bool Base::initializeSnapshotIfAppropriate(const unsigned int snapshotID)
{
	Snapshot* const snapshot = snapshots[snapshotID];

	if(snapshot->getSnapshotTime() >= 0)
		return true;

	if(snapshotID == 0)
	{
		updateSnapshot(snapshot);
		return true;
	}

	const FP previousSnapshotTime = snapshots[snapshotID - 1]->getSnapshotTime();

	if(previousSnapshotTime < 0)
		return false;

	const FP mpcTime = simulation->getMPCTime();
	const unsigned int snapshotCount = snapshots.size();
	const FP snapshotSeparation = measurementTime / snapshotCount;
	const FP deltaT = mpcTime - previousSnapshotTime;

	if(deltaT + 1e-6 > snapshotSeparation)
	{
		updateSnapshot(snapshot);
		return true;
	}

	return false;
}

void Base::measureWithSnapshot(Snapshot* const snapshot)
{
	const thrust::device_ptr<const MPCParticleVelocityType> snapshot_ptr =
		thrust::device_pointer_cast(snapshot->getBuffer());
	const thrust::device_ptr<const MPCParticleVelocityType> current_ptr =
		thrust::device_pointer_cast(currentVelocities);

	const MPCParticleVelocityType autocorrelationSum =
		thrust::inner_product(
			thrust::device,
			snapshot_ptr, snapshot_ptr + 3 * numberOfConstituents,
			current_ptr,
			MPCParticleVelocityType(0));

	const FP mpcTime = simulation->getMPCTime();
	const MPCParticleVelocityType autocorrelation = autocorrelationSum / numberOfConstituents;

	autocorrelations.push_back(boost::make_tuple(snapshot->getSnapshotTime(), mpcTime, autocorrelation));

	#ifdef OPENMPCD_DEBUG_CUDA_MPCFLUID_INSTRUMENTATION_VELOCITYAUTOCORRELATION
		const MPCParticleVelocityType naiveCPU = getCorrelationForSnapshot_CPU_naive(snapshot);
		const MPCParticleVelocityType thrustCPU = getCorrelationForSnapshot_CPU_thrust(snapshot);

		if(abs(autocorrelation - naiveCPU) > 1e-8 || abs(autocorrelation - thrustCPU) > 1e-8)
		{
			std::stringstream message;
			message<<"\nMISMATCHED RESULTS FOR VELOCITY AUTOCORRELATION.";
			message<<"\nSimulation Time: "<<mpcTime;
			message<<"\nSnapshot Time: "<<snapshot->getSnapshotTime();
			message<<"\nGPU Result (Thrust): "<<autocorrelation;
			message<<"\nCPU Result (naive) : "<<naiveCPU;
			message<<"\nCPU Result (Thrust): "<<thrustCPU;
			message<<"\n";
			OPENMPCD_THROW(Exception, message.str());
		}
	#endif
}

MPCParticleVelocityType Base::getCorrelationForSnapshot_CPU_naive(Snapshot* const snapshot)
{
	MPCParticleVelocityType* const current = new MPCParticleVelocityType[3 * numberOfConstituents];
	MPCParticleVelocityType* const snap    = new MPCParticleVelocityType[3 * numberOfConstituents];

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	cudaMemcpy(
		current, currentVelocities,
		3 * numberOfConstituents * sizeof(MPCParticleVelocityType), cudaMemcpyDeviceToHost);
	OPENMPCD_CUDA_THROW_ON_ERROR;

	cudaMemcpy(
		snap, snapshot->getBuffer(),
		3 * numberOfConstituents * sizeof(MPCParticleVelocityType), cudaMemcpyDeviceToHost);
	OPENMPCD_CUDA_THROW_ON_ERROR;

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	MPCParticleVelocityType sum = 0;
	for(unsigned int i=0; i < 3 * numberOfConstituents; ++i)
		sum += current[i] * snap[i];

	delete current;
	delete snap;

	return sum/numberOfConstituents;
}

MPCParticleVelocityType Base::getCorrelationForSnapshot_CPU_thrust(Snapshot* const snapshot)
{
	MPCParticleVelocityType* const current = new MPCParticleVelocityType[3 * numberOfConstituents];
	MPCParticleVelocityType* const snap    = new MPCParticleVelocityType[3 * numberOfConstituents];

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	cudaMemcpy(
		current, currentVelocities,
		3 * numberOfConstituents * sizeof(MPCParticleVelocityType), cudaMemcpyDeviceToHost);
	OPENMPCD_CUDA_THROW_ON_ERROR;

	cudaMemcpy(
		snap, snapshot->getBuffer(),
		3 * numberOfConstituents * sizeof(MPCParticleVelocityType), cudaMemcpyDeviceToHost);
	OPENMPCD_CUDA_THROW_ON_ERROR;

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	const MPCParticleVelocityType autocorrelationSum =
		thrust::inner_product(
			thrust::host,
			snap, snap + 3 * numberOfConstituents,
			current,
			MPCParticleVelocityType(0));

	delete current;
	delete snap;

	return autocorrelationSum/numberOfConstituents;
}

void Base::saveAutocorrelations(const std::string& path) const
{
	FilesystemUtilities::ensureParentDirectory(path);

	std::ofstream file(path.c_str(), std::ios::trunc);
	file.precision(std::numeric_limits<FP>::digits10 + 2);

	for(unsigned int i=0; i<autocorrelations.size(); ++i)
	{
		file
			<<autocorrelations[i].get<0>()<<"\t"
			<<autocorrelations[i].get<1>()<<"\t"
			<<autocorrelations[i].get<2>()<<"\n";
	}
}
