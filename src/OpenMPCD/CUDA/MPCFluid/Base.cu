#include <OpenMPCD/CUDA/MPCFluid/Base.hpp>

#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Base.hpp>
#include <OpenMPCD/CUDA/runtime.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>

using namespace OpenMPCD;

using OpenMPCD::CUDA::MPCFluid::Base;

Base::Base(
	const CUDA::Simulation* const sim, const unsigned int count, const FP streamingTimestep_,
	RNG& rng_, DeviceMemoryManager* const devMemMgr)
	: simulation(sim), deviceMemoryManager(devMemMgr), instrumentation(NULL),
	  numberOfParticles(count), streamingTimestep(streamingTimestep_)
{
	rng.seed(rng_());

	readConfiguration();

	mpcParticlePositions  = new MPCParticlePositionType[3 * getParticleCount()];
	mpcParticleVelocities = new MPCParticleVelocityType[3 * getParticleCount()];


	DeviceCode::setMPCParticleCountSymbol(getParticleCount());

	deviceMemoryManager->allocateMemory(
		&d_mpcParticlePositions,  3 * getParticleCount());
	deviceMemoryManager->allocateMemory(
		&d_mpcParticleVelocities, 3 * getParticleCount());
}

Base::~Base()
{
	delete instrumentation;

	delete[] mpcParticlePositions;
	delete[] mpcParticleVelocities;

	deviceMemoryManager->freeMemory(d_mpcParticlePositions);
	deviceMemoryManager->freeMemory(d_mpcParticleVelocities);
}

unsigned int Base::getNumberOfParticlesPerLogicalEntity() const
{
	OPENMPCD_THROW(InvalidCallException, "");
}

void Base::fetchFromDevice() const
{
	cudaMemcpy(mpcParticlePositions, d_mpcParticlePositions,
	           3 * sizeof(d_mpcParticlePositions[0])  * getParticleCount(),
	           cudaMemcpyDeviceToHost);
	OPENMPCD_CUDA_THROW_ON_ERROR;

	cudaMemcpy(mpcParticleVelocities, d_mpcParticleVelocities,
	           3 * sizeof(d_mpcParticlePositions[0])  * getParticleCount(),
	           cudaMemcpyDeviceToHost);
	OPENMPCD_CUDA_THROW_ON_ERROR;

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

const RemotelyStoredVector<const MPCParticlePositionType>
	Base::getPosition(const unsigned int particleID) const
{
	#ifdef OPENMPCD_DEBUG
		if(particleID >= getParticleCount())
			OPENMPCD_THROW(OutOfBoundsException, "particleID");
	#endif

	return RemotelyStoredVector<const MPCParticlePositionType>(mpcParticlePositions, particleID);
}

const RemotelyStoredVector<const MPCParticleVelocityType>
	Base::getVelocity(const unsigned int particleID) const
{
	#ifdef OPENMPCD_DEBUG
		if(particleID >= getParticleCount())
			OPENMPCD_THROW(OutOfBoundsException, "particleID");
	#endif

	return RemotelyStoredVector<const MPCParticleVelocityType>(mpcParticleVelocities, particleID);
}

void Base::setPositionsAndVelocities(
	const MPCParticlePositionType* const positions,
	const MPCParticleVelocityType* const velocities)
{
	#ifdef OPENMPCD_DEBUG
		if(positions == NULL)
			OPENMPCD_THROW(NULLPointerException, "`positions`");
		if(velocities == NULL)
			OPENMPCD_THROW(NULLPointerException, "`velocities`");
	#endif

	cudaMemcpy(d_mpcParticlePositions, positions,
	           3 * sizeof(d_mpcParticlePositions[0])  * getParticleCount(),
	           cudaMemcpyHostToDevice);
	OPENMPCD_CUDA_THROW_ON_ERROR;

	cudaMemcpy(d_mpcParticleVelocities, velocities,
	           3 * sizeof(d_mpcParticlePositions[0])  * getParticleCount(),
	           cudaMemcpyHostToDevice);
	OPENMPCD_CUDA_THROW_ON_ERROR;

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

void Base::writeToSnapshot(VTFSnapshotFile* const snapshot) const
{
	if(snapshot == NULL)
		OPENMPCD_THROW(NULLPointerException, "`snapshot`");

	if(snapshot->getNumberOfAtoms() != getParticleCount())
		OPENMPCD_THROW(InvalidArgumentException, "atom count");

	if(!snapshot->isInWriteMode())
		OPENMPCD_THROW(InvalidArgumentException, "`snapshot` not in write mode");

	fetchFromDevice();

	snapshot->writeTimestepBlock(mpcParticlePositions, mpcParticleVelocities);
}

void Base::findMatchingParticlesOnHost(
	bool (*func)(const RemotelyStoredVector<const MPCParticlePositionType>&, const RemotelyStoredVector<const MPCParticleVelocityType>&),
	std::vector<unsigned int>* const matches,
	unsigned int* const matchCount
	) const
{
	if(matchCount)
		*matchCount = 0;

	for(unsigned int pID = 0; pID < getParticleCount(); ++pID)
	{
		if(!func(getPosition(pID), getVelocity(pID)))
			continue;

		if(matches)
			matches->push_back(pID);

		if(matchCount)
			++(*matchCount);
	}
}

void Base::saveLogicalEntityCentersOfMassToDeviceMemory(
	MPCParticlePositionType* const buffer) const
{
	using
		OpenMPCD::CUDA::MPCFluid::DeviceCode::computeLogicalEntityCentersOfMass;

	if(!numberOfParticlesPerLogicalEntityIsConstant())
		OPENMPCD_THROW(UnimplementedException, "");

	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(buffer, OpenMPCD::NULLPointerException);

	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(getNumberOfLogicalEntities())
		computeLogicalEntityCentersOfMass<<<gridSize, blockSize>>>(
			workUnitOffset,
			getDevicePositions(),
			getNumberOfLogicalEntities(),
			getNumberOfParticlesPerLogicalEntity(),
			buffer);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

void Base::pushToDevice()
{
	cudaMemcpy(d_mpcParticlePositions, mpcParticlePositions,
	           3 * sizeof(d_mpcParticlePositions[0])  * getParticleCount(),
	           cudaMemcpyHostToDevice);
	OPENMPCD_CUDA_THROW_ON_ERROR;

	cudaMemcpy(d_mpcParticleVelocities, mpcParticleVelocities,
	           3 * sizeof(d_mpcParticlePositions[0])  * getParticleCount(),
	           cudaMemcpyHostToDevice);
	OPENMPCD_CUDA_THROW_ON_ERROR;

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

void Base::initializeVelocitiesOnHost() const
{
	const Configuration& config = simulation->getConfiguration();

	boost::random::normal_distribution<FP> velDist
		(config.read<FP>("initialization.particleVelocityDistribution.mean"),
		 config.read<FP>("initialization.particleVelocityDistribution.standardDeviation"));

	for(unsigned int pID = 0; pID < getParticleCount(); ++pID)
	{
		mpcParticleVelocities[pID * 3 + 0] = velDist(rng);
		mpcParticleVelocities[pID * 3 + 1] = velDist(rng);
		mpcParticleVelocities[pID * 3 + 2] = velDist(rng);
	}

	//make sure that the total momentum is 0
	const Vector3D<MPCParticleVelocityType> meanVelocity =
		getMeanMomentum(mpcParticleVelocities, getParticleCount()) /
		mpcParticleMass;

	for(unsigned int pID = 0; pID < getParticleCount(); ++pID)
	{
		mpcParticleVelocities[pID * 3 + 0] -= meanVelocity.getX();
		mpcParticleVelocities[pID * 3 + 1] -= meanVelocity.getY();
		mpcParticleVelocities[pID * 3 + 2] -= meanVelocity.getZ();
	}

	if(config.has("initialization.kT"))
	{
		globalUnbiasedThermostat(
			config.read<FP>("initialization.kT"),
			mpcParticleVelocities,
			getParticleCount());
	}
}

void Base::globalUnbiasedThermostat(const FP kT, MPCParticleVelocityType* const velocities,
	                                const unsigned int particleCount)
{
	const FP scaling=sqrt(kT/getkTViaKineticEnergy(velocities, particleCount));

	for(unsigned int i=0; i<particleCount; ++i)
	{
		for(unsigned int coordinate=0; coordinate<3; ++coordinate)
			velocities[3 * i + coordinate] *= scaling;
	}
}

const Vector3D<MPCParticleVelocityType>
	Base::getTotalMomentum(const MPCParticleVelocityType* const velocities,
	                          const unsigned int particleCount)
{
	Vector3D<MPCParticleVelocityType> ret(0, 0, 0);

	for(unsigned int i=0; i<particleCount; ++i)
	{
		const RemotelyStoredVector<const MPCParticleVelocityType> velocity(velocities, i);

		ret += velocity;
	}

	return ret * mpcParticleMass;
}

const Vector3D<MPCParticleVelocityType>
	Base::getMeanMomentum(const MPCParticleVelocityType* const velocities,
	                         const unsigned int particleCount)
{
	return getTotalMomentum(velocities, particleCount) / particleCount;
}

FP Base::getKineticEnergy(const MPCParticleVelocityType* const velocities,
                             const unsigned int particleCount)
{
	FP ret=0;

	for(unsigned int i=0; i<particleCount; ++i)
	{
		for(unsigned int coordinate=0; coordinate<3; ++coordinate)
			ret += velocities[3 * i + coordinate] * velocities[3 * i + coordinate];
	}

	return mpcParticleMass*ret/2.0;
}

void Base::readConfiguration()
{
}
