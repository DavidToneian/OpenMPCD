#include <OpenMPCD/CUDA/MPCFluid/GaussianDumbbells.hpp>

#include <OpenMPCD/AnalyticQuantitiesGaussianDumbbell.hpp>
#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/CUDA/BoundaryCondition/LeesEdwards.hpp>
#include <OpenMPCD/CUDA/DeviceCode/LeesEdwardsBoundaryConditions.hpp>
#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/GaussianDumbbells.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/GaussianDumbbells.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>
#include <OpenMPCD/Exceptions.hpp>

using namespace OpenMPCD;
using namespace OpenMPCD::CUDA;
using namespace OpenMPCD::CUDA::MPCFluid;

GaussianDumbbells::GaussianDumbbells(const CUDA::Simulation* const sim, const unsigned int count,
                                     const FP streamingTimestep_, RNG& rng_,
                                     DeviceMemoryManager* const devMemMgr,
                                     const FP kT, const FP leesEdwardsShearRate)
	: Base(sim, count, streamingTimestep_, rng_, devMemMgr)
{
	if(
		dynamic_cast<const BoundaryCondition::LeesEdwards*>(
			simulation->getBoundaryConditions())
		== NULL )
	{
		OPENMPCD_THROW(
			UnimplementedException,
			"Currently, only Lees-Edwards boundary conditions are "
			"supported.");
	}

	static bool firstInvocation = true;
	if(!firstInvocation)
		OPENMPCD_THROW(InvalidCallException, "Cannot construct more than one instance of "
		                                    "OpenMPCD::CUDA::GaussianDumbbells, "
		                                    "since it uses a CUDA Device constant.");

	if(getParticleCount() % 2 != 0)
		OPENMPCD_THROW(InvalidConfigurationException, "For dumbbell simulations, the total number of "
		                                             "particles has to be even.");

	readConfiguration();

	initializeOnHost(leesEdwardsShearRate);

	{
		//When given a zero-shear relaxation time `tau_0` in the configuration
		//(key `"zeroShearRelaxationTime"`), one needs to compute the
		//frequency `omega` or the spring constant. In either case, it is
		//assumed that there is some well-defined temperature `T`
		//that gives, when multiplied with Boltzmann's constant, `kT`.
		//As such, assert that there is some kind of effective bulk thermostat:
		simulation->getConfiguration().assertValue(
			"bulkThermostat.type", "MBS"); //currently the only one implemented

		const FP m      = mpcParticleMass;
		const FP l      = dumbbellRootMeanSquareLength;
		const FP tau_0  = zeroShearRelaxationTime;
		const FP lambda = AnalyticQuantitiesGaussianDumbbell::lagrangianMultiplier(l, leesEdwardsShearRate, tau_0);

		reducedSpringConstant = 2 * lambda * kT / m;

		if(streamAnalyticallyFlag)
		{
			const FP omega = sqrt(4 * lambda * kT / m);
			DeviceCode::setGaussianDumbbellSymbols(omega, streamingTimestep);
		}
	}

	pushToDevice();

	instrumentation = new Instrumentation::GaussianDumbbells(sim, devMemMgr, this);
}

void GaussianDumbbells::readConfiguration()
{
	const Configuration& config = simulation->getConfiguration();

	config.read("mpc.fluid.dumbbell.analyticalStreaming",     &streamAnalyticallyFlag);
	config.read("mpc.fluid.dumbbell.rootMeanSquareLength",    &dumbbellRootMeanSquareLength);
	config.read("mpc.fluid.dumbbell.zeroShearRelaxationTime", &zeroShearRelaxationTime);

	if(!streamAnalyticallyFlag)
		config.read("mpc.fluid.dumbbell.mdStepCount", &mdStepCount);
}

void GaussianDumbbells::stream()
{
	if(streamAnalyticallyFlag)
	{
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(getParticleCount() / 2)
			DeviceCode::streamDumbbellsAnalytically<<<gridSize, blockSize>>>(
				workUnitOffset,
				d_mpcParticlePositions,
				d_mpcParticleVelocities);
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	}
	else
	{
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(getParticleCount() / 2)
			DeviceCode::streamDumbbellsVelocityVerlet<<<gridSize, blockSize>>>(
				workUnitOffset,
				d_mpcParticlePositions,
				d_mpcParticleVelocities,
				reducedSpringConstant,
				streamingTimestep / mdStepCount,
				mdStepCount);
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	}

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

void GaussianDumbbells::initializeOnHost(const FP leesEdwardsShearRate)
{
	initializeVelocitiesOnHost();


	const FP tau_0  = zeroShearRelaxationTime;
	const FP Wi     = AnalyticQuantitiesGaussianDumbbell::weissenbergNumber(leesEdwardsShearRate, tau_0);

	boost::random::uniform_01<FP> posDist;

	for(unsigned int pID = 0; pID < getParticleCount(); pID += 2)
	{
		mpcParticlePositions[pID * 3 + 0] = posDist(rng) * simulation->getSimulationBoxSizeX();
		mpcParticlePositions[pID * 3 + 1] = posDist(rng) * simulation->getSimulationBoxSizeY();
		mpcParticlePositions[pID * 3 + 2] = posDist(rng) * simulation->getSimulationBoxSizeZ();

		const RemotelyStoredVector<MPCParticlePositionType> position1(mpcParticlePositions, pID);

		const Vector3D<MPCParticlePositionType> position2 = getInitialDumbbellPartnerPosition(position1, Wi);

		if(	position2.getX() >= simulation->getSimulationBoxSizeX() ||
			position2.getY() >= simulation->getSimulationBoxSizeY() ||
			position2.getZ() >= simulation->getSimulationBoxSizeZ() ||
			position2.getX() < 0 || position2.getY() < 0 || position2.getZ() < 0)
		{
			//The second particle would start outside of the box,
			//thus gaining in velocity under Lees-Edwards boundary conditions.
			//Choose new positions instead.
			pID-=2;
			continue;
		}


		mpcParticlePositions[(pID+1) * 3 + 0] = position2.getX();
		mpcParticlePositions[(pID+1) * 3 + 1] = position2.getY();
		mpcParticlePositions[(pID+1) * 3 + 2] = position2.getZ();
	}
}

const Vector3D<MPCParticlePositionType>
	GaussianDumbbells::getInitialDumbbellPartnerPosition(
		const RemotelyStoredVector<MPCParticlePositionType>& position1,
		const FP Wi) const
{
	const Configuration& config = simulation->getConfiguration();

	static const std::string positioning
		= config.read<std::string>("initialization.dumbbell.relativePosition");

	if(positioning=="relaxed")
	{
		const Vector3D<MPCParticlePositionType> displacement
				= Vector3D<MPCParticlePositionType>::getRandomUnitVector(rng) * dumbbellRootMeanSquareLength;
		return position1 + displacement;
	}

	if(positioning=="perfect")
	{
		static const FP mu     = AnalyticQuantitiesGaussianDumbbell::lagrangianMultiplierRatio(Wi);

		static const FP prefactor = dumbbellRootMeanSquareLength * dumbbellRootMeanSquareLength / (3.0 * mu);

		static const FP R_x = sqrt(prefactor * (1 + Wi * Wi / (2.0 * mu * mu)));
		static const FP R_y = sqrt(prefactor);
		static const FP R_z = sqrt(prefactor);
		static const Vector3D<MPCParticlePositionType> displacement(R_x, R_y, R_z);

		return position1 + displacement;
	}

	OPENMPCD_THROW(InvalidConfigurationException, "initialization.dumbbell.relativePosition has an invalid value.");
}
