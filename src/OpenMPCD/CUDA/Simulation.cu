#include <OpenMPCD/CUDA/Simulation.hpp>

#include <OpenMPCD/CUDA/BoundaryCondition/Factory.hpp>
#include <OpenMPCD/CUDA/DeviceCode/LeesEdwardsBoundaryConditions.hpp>
#include <OpenMPCD/CUDA/DeviceCode/Simulation.hpp>
#include <OpenMPCD/CUDA/DeviceCode/Symbols.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Factory.hpp>
#include <OpenMPCD/CUDA/MPCSolute/Factory.hpp>
#include <OpenMPCD/CUDA/runtime.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>
#include <OpenMPCD/Profiling/CodeRegion.hpp>
#include <OpenMPCD/RemotelyStoredVector.hpp>

#include <boost/math/special_functions/round.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int_distribution.hpp>

using namespace OpenMPCD;

CUDA::Simulation::Simulation(const std::string& configurationFilename,
                             const unsigned int rngSeed, const std::string& dir)
	: config(configurationFilename), numberOfCompletedSweeps(0), rng(rngSeed),
	  gpurngs(NULL),
	  mpcTime(0), rundir(dir),
	  mpcFluid(NULL), mpcSolute(NULL), boundaryCondition(NULL)
{
	readConfiguration();

	initialize();
}

CUDA::Simulation::Simulation(const Configuration& configuration,
                             const unsigned int rngSeed)
	: config(configuration), numberOfCompletedSweeps(0),
	  rng(rngSeed), gpurngs(NULL),
	  mpcTime(0), mpcFluid(NULL), mpcSolute(NULL),
	  boundaryCondition(NULL)
{
	readConfiguration();

	initialize();
}

CUDA::Simulation::~Simulation()
{
	delete mpcFluid;
	delete mpcSolute;
	delete boundaryCondition;

	DeviceCode::destroyGPURNGs<<<1, 1>>>(getCollisionCellCount(), gpurngs);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
	deviceMemoryManager.freeMemory(gpurngs);

	deviceMemoryManager.freeMemory(d_gridShift);
	deviceMemoryManager.freeMemory(d_leesEdwardsVelocityShift);
	deviceMemoryManager.freeMemory(d_fluidCollisionCellIndices);
	deviceMemoryManager.freeMemory(d_collisionCellParticleCounts);
	deviceMemoryManager.freeMemory(d_collisionCellMomenta);

	delete[] collisionCellRotationAxes;

	deviceMemoryManager.freeMemory(d_collisionCellRotationAxes);

	delete[] collisionCellRelativeVelocityScalings;
	deviceMemoryManager.freeMemory(d_collisionCellRelativeVelocityScalings);

	deviceMemoryManager.freeMemory(d_collisionCellFrameInternalKineticEnergies);
	deviceMemoryManager.freeMemory(d_collisionCellMasses);

	deviceMemoryManager.freeMemory(d_leesEdwardsVelocityShift_solute);
	deviceMemoryManager.freeMemory(d_collisionCellIndices_solute);
}

void CUDA::Simulation::warmup()
{
	run(config.read<unsigned int>("mpc.warmupSteps"));
}

void CUDA::Simulation::sweep()
{
	run(mpcSweepSize);


	++numberOfCompletedSweeps;

	#ifdef OPENMPCD_DEBUG
		if(numberOfCompletedSweeps == 0)
			OPENMPCD_THROW(Exception, "Overflow of `numberOfCompletedSweeps`");
	#endif
}

unsigned int CUDA::Simulation::getNumberOfCompletedSweeps() const
{
	return numberOfCompletedSweeps;
}

void CUDA::Simulation::readConfiguration()
{
	config.read("mpc.simulationBoxSize.x", &mpcSimulationBoxSizeX);
	config.read("mpc.simulationBoxSize.y", &mpcSimulationBoxSizeY);
	config.read("mpc.simulationBoxSize.z", &mpcSimulationBoxSizeZ);
	config.read("mpc.timestep",          &mpcTimestep);
	config.read("mpc.srdCollisionAngle", &srdCollisionAngle);
	config.read("mpc.gridShiftScale",    &gridShiftScale);

	bulkThermostatTargetkT = 0;
	if(config.has("bulkThermostat"))
		config.read("bulkThermostat.targetkT", &bulkThermostatTargetkT);

	config.read("mpc.sweepSize", &mpcSweepSize);
}

void CUDA::Simulation::initialize()
{
	const unsigned int collisionCellCount = getCollisionCellCount();

	const unsigned int mpcParticleDensity =
		config.read<unsigned int>("initialization.particleDensity");

	if(!config.has("boundaryConditions"))
	{
		OPENMPCD_THROW(
			InvalidConfigurationException,
			"No boundary conditions configured.");
	}

	boundaryCondition =
		BoundaryCondition::Factory::getInstance(
			config.getSetting("boundaryConditions"),
			getSimulationBoxSizeX(),
			getSimulationBoxSizeY(),
			getSimulationBoxSizeZ());

	const unsigned int mpcParticleCount =
		static_cast<unsigned int>(
			boost::math::iround(
				mpcParticleDensity *
				boundaryCondition->getTotalAvailableVolume()));

	mpcFluid = MPCFluid::Factory::getInstance(this, config, mpcParticleCount, rng);

	if(mpcFluid)
	{
		if(mpcParticleDensity == 0)
		{
			OPENMPCD_THROW(
				InvalidConfigurationException,
				"`initialization.particleDensity` must be `0` if and only if"
				"`mpc.fluid` does not exist.");
		}

		if(mpcParticleCount == 0)
		{
			OPENMPCD_THROW(
				InvalidConfigurationException,
				"`initialization.particleDensity` is not `0`, but boundary "
				"conditions are such that no fluid particles are created.");
		}
	}
	else
	{
		if(mpcParticleDensity != 0)
		{
			OPENMPCD_THROW(
				InvalidConfigurationException,
				"`initialization.particleDensity` must be `0` if and only if"
				"`mpc.fluid` does not exist.");
		}
	}


	boost::random::uniform_int_distribution<unsigned long long> seedDist;
	deviceMemoryManager.allocateMemory(&gpurngs, collisionCellCount);
	DeviceCode::constructGPURNGs<<<1, 1>>>(
		collisionCellCount, gpurngs, seedDist(rng));
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	DeviceCode::setSimulationBoxSizeSymbols(mpcSimulationBoxSizeX, mpcSimulationBoxSizeY, mpcSimulationBoxSizeZ);
	DeviceCode::setMPCStreamingTimestep(mpcTimestep);
	DeviceCode::setSRDCollisionAngleSymbol(srdCollisionAngle);

	deviceMemoryManager.allocateMemory(&d_gridShift, 3);
	deviceMemoryManager.allocateMemory(&d_leesEdwardsVelocityShift, mpcParticleCount);
	deviceMemoryManager.allocateMemory(&d_fluidCollisionCellIndices, mpcParticleCount);
	deviceMemoryManager.allocateMemory(&d_collisionCellParticleCounts, collisionCellCount);
	deviceMemoryManager.allocateMemory(&d_collisionCellMomenta, 3 * collisionCellCount);
	deviceMemoryManager.allocateMemory(
		&d_collisionCellFrameInternalKineticEnergies,
		collisionCellCount);
	deviceMemoryManager.allocateMemory(&d_collisionCellMasses, collisionCellCount);

	collisionCellRotationAxes = new MPCParticlePositionType[3*collisionCellCount];

	deviceMemoryManager.allocateMemory(&d_collisionCellRotationAxes, 3 * collisionCellCount);

	collisionCellRelativeVelocityScalings = new FP[collisionCellCount];
	deviceMemoryManager.allocateMemory(&d_collisionCellRelativeVelocityScalings, collisionCellCount);
	//initialize to 1, because the scaling will be done whether or not the MBS
	//thermostat is configured (and generate `generateCollisionCellMBSFactors`
	//does anything):
	for(unsigned int i = 0; i < collisionCellCount; ++i)
		collisionCellRelativeVelocityScalings[i] = 1;
	deviceMemoryManager.copyElementsFromHostToDevice(
		collisionCellRelativeVelocityScalings,
		d_collisionCellRelativeVelocityScalings,
		collisionCellCount);


	d_leesEdwardsVelocityShift_solute = NULL;
	d_collisionCellIndices_solute = NULL;

	mpcSolute = MPCSolute::Factory::getInstance(this, config, rng);
	if(mpcSolute)
	{
		const FP mdTimeStepSize = mpcSolute->getMDTimeStepSize();
		const std::size_t mdStepCount = mpcTimestep / mdTimeStepSize;

		if(mdTimeStepSize * mdStepCount != mpcTimestep)
		{
			OPENMPCD_THROW(
				InvalidConfigurationException,
				"The MPC solute MD step size must be chosen such that the MPC "
				"streaming timestep size is an integer multiple of the MD step "
				"size");
		}

		deviceMemoryManager.allocateMemory(
			&d_leesEdwardsVelocityShift_solute,
			mpcSolute->getParticleCount());

		deviceMemoryManager.allocateMemory(
			&d_collisionCellIndices_solute,
			mpcSolute->getParticleCount());
	}
}

void CUDA::Simulation::run(const unsigned int stepCount)
{
	for(unsigned int i=0; i<stepCount; ++i)
	{
		stream();
		collide();
	}

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

void CUDA::Simulation::stream()
{
	const Profiling::CodeRegion codeRegion("stream");

	if(mpcFluid)
		mpcFluid->stream();
	if(mpcSolute)
	{
		const FP mdTimeStepSize = mpcSolute->getMDTimeStepSize();
		const std::size_t mdStepCount = mpcTimestep / mdTimeStepSize;

		OPENMPCD_DEBUG_ASSERT(mdTimeStepSize * mdStepCount == mpcTimestep);

		for(std::size_t i = 0; i< mdStepCount; ++i)
			mpcSolute->performMDTimestep();
	}

	mpcTime += mpcTimestep;
}

void CUDA::Simulation::collide()
{
	if(!hasMPCFluid() && !hasSolute())
		return;

	if(srdCollisionAngle == 0 && bulkThermostatTargetkT == 0)
		return;

	const Profiling::CodeRegion codeRegion("collide");

	const unsigned int collisionCellCount = getCollisionCellCount();

	generateGridShiftVector();
	generateCollisionCellRotationAxes();

	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(collisionCellCount)
		DeviceCode::resetCollisionCellData
			<<<gridSize, blockSize>>> (
				workUnitOffset,
				collisionCellCount,
				d_collisionCellParticleCounts,
				d_collisionCellMomenta,
				d_collisionCellFrameInternalKineticEnergies,
				d_collisionCellMasses);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END


	if(hasMPCFluid())
	{
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(mpcFluid->getParticleCount())
			DeviceCode::sortParticlesIntoCollisionCellsLeesEdwards
				<<<gridSize, blockSize>>> (
					workUnitOffset,
					mpcFluid->getParticleCount(),
					d_gridShift,
					mpcTime,
					mpcFluid->getDevicePositions(),
					mpcFluid->getDeviceVelocities(),
					d_leesEdwardsVelocityShift,
					d_fluidCollisionCellIndices);
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	}
	if(hasSolute())
	{
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(mpcSolute->getParticleCount())
			DeviceCode::sortParticlesIntoCollisionCellsLeesEdwards
				<<<gridSize, blockSize>>> (
					workUnitOffset,
					mpcSolute->getParticleCount(),
					d_gridShift,
					mpcTime,
					mpcSolute->getDevicePositions(),
					mpcSolute->getDeviceVelocities(),
					d_leesEdwardsVelocityShift_solute,
					d_collisionCellIndices_solute);
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	}



	if(hasMPCFluid())
	{
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(mpcFluid->getParticleCount())
			DeviceCode::collisionCellContributions <<<gridSize, blockSize>>> (
				workUnitOffset,
				mpcFluid->getParticleCount(),
				mpcFluid->getDeviceVelocities(),
				d_fluidCollisionCellIndices,
				d_collisionCellMomenta,
				d_collisionCellMasses,
				mpcFluid->getParticleMass());
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	}
	if(hasSolute())
	{
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(mpcSolute->getParticleCount())
			DeviceCode::collisionCellContributions <<<gridSize, blockSize>>> (
				workUnitOffset,
				mpcSolute->getParticleCount(),
				mpcSolute->getDeviceVelocities(),
				d_collisionCellIndices_solute,
				d_collisionCellMomenta,
				d_collisionCellMasses,
				mpcSolute->getParticleMass());
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	}



	if(hasMPCFluid())
	{
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(mpcFluid->getParticleCount())
			DeviceCode::collisionCellStochasticRotationStep1
				<<<gridSize, blockSize>>> (
					workUnitOffset,
					mpcFluid->getParticleCount(),
					mpcFluid->getDeviceVelocities(),
					mpcFluid->getParticleMass(),
					d_fluidCollisionCellIndices,
					d_collisionCellMomenta,
					d_collisionCellMasses,
					d_collisionCellRotationAxes,
					d_collisionCellFrameInternalKineticEnergies,
					d_collisionCellParticleCounts);
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	}
	if(hasSolute())
	{
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(mpcSolute->getParticleCount())
			DeviceCode::collisionCellStochasticRotationStep1
				<<<gridSize, blockSize>>> (
					workUnitOffset,
					mpcSolute->getParticleCount(),
					mpcSolute->getDeviceVelocities(),
					mpcSolute->getParticleMass(),
					d_collisionCellIndices_solute,
					d_collisionCellMomenta,
					d_collisionCellMasses,
					d_collisionCellRotationAxes,
					d_collisionCellFrameInternalKineticEnergies,
					d_collisionCellParticleCounts);
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	}


	generateCollisionCellMBSFactors();


	if(hasMPCFluid())
	{
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(mpcFluid->getParticleCount())
			DeviceCode::collisionCellStochasticRotationStep2
				<<<gridSize, blockSize>>> (
					workUnitOffset,
					mpcFluid->getParticleCount(),
					mpcFluid->getDeviceVelocities(),
					d_fluidCollisionCellIndices,
					d_collisionCellMomenta,
					d_collisionCellMasses,
					d_collisionCellRelativeVelocityScalings);
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	}
	if(hasSolute())
	{
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(mpcSolute->getParticleCount())
			DeviceCode::collisionCellStochasticRotationStep2
				<<<gridSize, blockSize>>> (
					workUnitOffset,
					mpcSolute->getParticleCount(),
					mpcSolute->getDeviceVelocities(),
					d_collisionCellIndices_solute,
					d_collisionCellMomenta,
					d_collisionCellMasses,
					d_collisionCellRelativeVelocityScalings);
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	}



	if(hasMPCFluid())
	{
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(mpcFluid->getParticleCount())
			DeviceCode::undoLeesEdwardsVelocityCorrections
				<<<gridSize, blockSize>>> (
					workUnitOffset,
					mpcFluid->getParticleCount(),
					mpcFluid->getDeviceVelocities(),
					d_leesEdwardsVelocityShift);
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	}
	if(hasSolute())
	{
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(mpcSolute->getParticleCount())
			DeviceCode::undoLeesEdwardsVelocityCorrections
				<<<gridSize, blockSize>>> (
					workUnitOffset,
					mpcSolute->getParticleCount(),
					mpcSolute->getDeviceVelocities(),
					d_leesEdwardsVelocityShift_solute);
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	}
}

void CUDA::Simulation::generateGridShiftVector()
{
	DeviceCode::generateGridShiftVector<<<1, 1>>>(
		d_gridShift,
		gridShiftScale,
		gpurngs);
}

void CUDA::Simulation::generateCollisionCellRotationAxes()
{
	const unsigned int collisionCellCount = getCollisionCellCount();

	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(collisionCellCount)
		DeviceCode::generateCollisionCellRotationAxes
			<<<gridSize, blockSize>>> (
				workUnitOffset,
				collisionCellCount,
				d_collisionCellRotationAxes,
				gpurngs);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
}

void CUDA::Simulation::generateCollisionCellMBSFactors()
{
	if(bulkThermostatTargetkT == 0)
		return;

	const unsigned int collisionCellCount = getCollisionCellCount();

	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(collisionCellCount)
		DeviceCode::generateCollisionCellMBSFactors
			<<<gridSize, blockSize>>> (
				workUnitOffset,
				collisionCellCount,
				d_collisionCellFrameInternalKineticEnergies,
				d_collisionCellParticleCounts,
				d_collisionCellRelativeVelocityScalings,
				bulkThermostatTargetkT,
				gpurngs);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
}
