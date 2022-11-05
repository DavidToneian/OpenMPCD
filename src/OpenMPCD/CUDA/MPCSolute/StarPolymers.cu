#include <OpenMPCD/CUDA/MPCSolute/StarPolymers.hpp>

#include <OpenMPCD/CUDA/MPCSolute/ImplementationDetails/StarPolymers.hpp>

#include <OpenMPCD/CUDA/DeviceCode/VelocityVerlet.hpp>
#include <OpenMPCD/CUDA/DeviceCode/Utilities.hpp>
#include <OpenMPCD/CUDA/MPCSolute/Instrumentation/StarPolymers.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>
#include <OpenMPCD/Utility/CompilerDetection.hpp>

#include <boost/math/constants/constants.hpp>

#include <iostream>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCSolute
{

namespace ImplementationDetails
{
namespace StarPolymers
{

/**
 * Updates the list of force vectors by iterating over all interactions of the
 * fixed `particleID1` and the variable `particleID2`, which ranges from
 * `particleID1 + 1` to `totalNumberOfParticles - 1`, inclusive. The forces are
 * updated not only for `particleID1`, but for all particles involved in the
 * interactions considered.
 *
 * @tparam VelocityCoordinate
 *         The type of the velocity coordinates.
 * @tparam ForceCoordinate
 *         The type of the force coordinates.
 * @tparam PositionCoordinate
 *         The type of the position coordinates.
 *
 * @param[in]     particleID1
 *                The ID of the first particle interacting.
 * @param[in]     positions
 *                The position coordinates.
 * @param[in,out] forces
 *                The force coordinates.
 * @param[in]     position1
 *                The position of the first particle.
 * @param[in]     position2
 *                The position of the second particle.
 * @param[in]     armCountPerStar
 *                The number of polymer arms per star.
 * @param[in]     particleCountPerArm
 *                The number of non-magnetic particles per arm.
 * @param[in]     hasMagneticParticles
 *                Whether each arm has an additional, magnetic particle
 *                attached.
 * @param[in]     WCAPotentials
 *                The array of allocated WCA potentials.
 * @param[in]     FENEPotentials
 *                The array of allocated FENE potentials.
 * @param[in]     magneticPotential
 *                The allocated magnetic potential.
 * @param[in]     workUnitOffset
 *                The number of particles to skip as the interaction partner of
 *                `particleID1`.
 */
template<
	typename VelocityCoordinate,
	typename ForceCoordinate,
	typename PositionCoordinate>
__global__ void
updateForceBuffer(
	const std::size_t particleID1,
	const PositionCoordinate* const positions,
	ForceCoordinate* const forces,
	const std::size_t starCount,
	const std::size_t armCountPerStar,
	const std::size_t particleCountPerArm,
	const bool hasMagneticParticles,
	PairPotentials::
		WeeksChandlerAndersen_DistanceOffset<ForceCoordinate>** const
			wcaPotentials,
	PairPotentials::FENE<ForceCoordinate>** const fenePotentials,
	PairPotentials::
		MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<
			ForceCoordinate>** const
				magneticPotential,
	const std::size_t workUnitOffset)
{
	const std::size_t particleID2 =
		particleID1 + 1 + blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;

	const std::size_t totalParticleCount =
		getParticleCount(
			starCount, armCountPerStar, particleCountPerArm,
			hasMagneticParticles);

	if(particleID2 >= totalParticleCount)
		return;

	const Vector3D<ForceCoordinate> forceOnParticle1DueToParticle2 =
		computeForceOnParticle1DueToParticle2(
			particleID1,
			particleID2,
			RemotelyStoredVector<const PositionCoordinate>(
				positions, particleID1),
			RemotelyStoredVector<const PositionCoordinate>(
				positions, particleID2),
			starCount,
			armCountPerStar,
			particleCountPerArm,
			hasMagneticParticles,
			wcaPotentials,
			fenePotentials,
			magneticPotential);


	//RemotelyStoredVector<ForceCoordinate> forceOnParticle1(forces, particleID1);
	OpenMPCD::CUDA::DeviceCode::atomicAdd(&forces[particleID1 * 3 + 0], forceOnParticle1DueToParticle2.getX());
	OpenMPCD::CUDA::DeviceCode::atomicAdd(&forces[particleID1 * 3 + 1], forceOnParticle1DueToParticle2.getY());
	OpenMPCD::CUDA::DeviceCode::atomicAdd(&forces[particleID1 * 3 + 2], forceOnParticle1DueToParticle2.getZ());
	//forceOnParticle1.atomicAdd(forceOnParticle1DueToParticle2);

	//RemotelyStoredVector<ForceCoordinate> forceOnParticle2(forces, particleID2);
	OpenMPCD::CUDA::DeviceCode::atomicAdd(&forces[particleID2 * 3 + 0], -forceOnParticle1DueToParticle2.getX());
	OpenMPCD::CUDA::DeviceCode::atomicAdd(&forces[particleID2 * 3 + 1], -forceOnParticle1DueToParticle2.getY());
	OpenMPCD::CUDA::DeviceCode::atomicAdd(&forces[particleID2 * 3 + 2], -forceOnParticle1DueToParticle2.getZ());
	//forceOnParticle2.atomicAdd(-forceOnParticle1DueToParticle2);
}


template<
	typename PositionCoordinate,
	typename VelocityCoordinate,
	typename ForceCoordinate
	>
__global__ void velocityVerletKernel1(
	PositionCoordinate* const positions,
	const VelocityCoordinate* const velocities,
	const ForceCoordinate* const forces,
	const FP timestep,
	const std::size_t particleCount,
	const FP mass,
	const std::size_t workUnitOffset)
{
	const std::size_t coordinateID =
		blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;

	if(coordinateID >= 3 * particleCount)
		return;

	positions[coordinateID] =
		OpenMPCD::CUDA::DeviceCode::velocityVerletStep1(
			positions[coordinateID],
			velocities[coordinateID],
			forces[coordinateID] / mass,
			timestep);
}

template<
	typename VelocityCoordinate,
	typename ForceCoordinate
	>
__global__ void velocityVerletKernel2(
	VelocityCoordinate* const velocities,
	const ForceCoordinate* const oldForces,
	const ForceCoordinate* const newForces,
	const FP timestep,
	const std::size_t particleCount,
	const FP mass,
	const std::size_t workUnitOffset)
{
	const std::size_t coordinateID =
		blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;

	if(coordinateID >= 3 * particleCount)
		return;

	velocities[coordinateID] =
		OpenMPCD::CUDA::DeviceCode::velocityVerletStep2(
			velocities[coordinateID],
			oldForces[coordinateID] / mass,
			newForces[coordinateID] / mass,
			timestep);
}

} //namespace StarPolymers
} //namespace ImplementationDetails


template<typename PositionCoordinate, typename VelocityCoordinate>
StarPolymers<PositionCoordinate, VelocityCoordinate>::StarPolymers(
	const Configuration::Setting& settings,
	const BoundaryCondition::Base* const boundaryCondition)
	: particleMass(1)
{
	if(
		dynamic_cast<const BoundaryCondition::LeesEdwards*>(boundaryCondition)
		== NULL )
	{
		OPENMPCD_THROW(
			UnimplementedException,
			"Currently, only Lees-Edwards boundary conditions are "
			"supported.");
	}

	deviceMemoryManager.setAutofree(true);

	settings.read("mdTimeStepSize", &mdTimeStepSize);
	if(mdTimeStepSize <= 0)
		OPENMPCD_THROW(InvalidConfigurationException, "`mdTimeStepSize`");

	settings.read("structure.starCount",            &starCount);
	settings.read("structure.armCountPerStar",      &armCountPerStar);
	settings.read("structure.armParticlesPerArm",   &armParticlesPerArm);
	settings.read("structure.hasMagneticParticles", &hasMagneticParticles_);

	if(settings.has("structure.particleMass"))
		settings.read("structure.particleMass", &particleMass);

	const std::size_t coordinateCount = 3 * getParticleCount();

	OPENMPCD_DEBUG_ASSERT(coordinateCount != 0);

	deviceMemoryManager.allocateMemory(&d_positions, coordinateCount);
	deviceMemoryManager.allocateMemory(&d_velocities, coordinateCount);
	deviceMemoryManager.allocateMemory(&d_forces1, coordinateCount);
	deviceMemoryManager.allocateMemory(&d_forces2, coordinateCount);

	#ifdef OPENMPCD_COMPILER_GCC
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Wvla"
	#endif
	h_positions.reset(new PositionCoordinate[coordinateCount]);
	h_velocities.reset(new VelocityCoordinate[coordinateCount]);
	#ifdef OPENMPCD_COMPILER_GCC
		#pragma GCC diagnostic pop
	#endif

	ForceCoordinate epsilon_core, epsilon_arm, epsilon_magnetic;
	ForceCoordinate sigma_core, sigma_arm, sigma_magnetic;
	ForceCoordinate D_core, D_arm, D_magnetic;
	ForceCoordinate magneticPrefactor;

	settings.read("interactionParameters.epsilon_core",     &epsilon_core);
	settings.read("interactionParameters.epsilon_arm",      &epsilon_arm);
	settings.read("interactionParameters.epsilon_magnetic", &epsilon_magnetic);

	settings.read("interactionParameters.sigma_core",     &sigma_core);
	settings.read("interactionParameters.sigma_arm",      &sigma_arm);
	settings.read("interactionParameters.sigma_magnetic", &sigma_magnetic);

	settings.read("interactionParameters.D_core",     &D_core);
	settings.read("interactionParameters.D_arm",      &D_arm);
	settings.read("interactionParameters.D_magnetic", &D_magnetic);

	settings.read(
		"interactionParameters.magneticPrefactor", &magneticPrefactor);

	Vector3D<ForceCoordinate> magneticDipoleOrientation(0, 0, 1);
	if(settings.has("interactionParameters.dipoleOrientation"))
	{
		magneticDipoleOrientation.set(
			settings.getList("interactionParameters.dipoleOrientation").
				read<ForceCoordinate>(0),
			settings.getList("interactionParameters.dipoleOrientation").
				read<ForceCoordinate>(1),
			settings.getList("interactionParameters.dipoleOrientation").
				read<ForceCoordinate>(2)
			);
	}
	magneticDipoleOrientation.normalize();

	ImplementationDetails::StarPolymers::createInteractionsOnDevice(
		epsilon_core, epsilon_arm, epsilon_magnetic,
		sigma_core, sigma_arm, sigma_magnetic,
		D_core, D_arm, D_magnetic,
		magneticPrefactor, magneticDipoleOrientation,
		&wcaPotentials, &fenePotentials, &magneticPotential
		);


	if(settings.has("initialState"))
	{
		for(std::size_t i = 0; i < coordinateCount; ++i)
			h_velocities[i] = 0;

		VTFSnapshotFile inputSnapshot(
			settings.read<std::string>("initialState"));
		readStateFromSnapshot(&inputSnapshot);
	}
	else
	{
		initializePositionsOnHost();
		initializeVelocitiesOnHost();
	}

	pushToDevice();

	this->instrumentation =
		new	Instrumentation::StarPolymers<
			PositionCoordinate, VelocityCoordinate>(
				this, settings);
}

template<typename PositionCoordinate, typename VelocityCoordinate>
StarPolymers<PositionCoordinate, VelocityCoordinate>::~StarPolymers()
{
	ImplementationDetails::StarPolymers::destroyInteractionsOnDevice(
		wcaPotentials, fenePotentials, magneticPotential);
}

template<typename PositionCoordinate, typename VelocityCoordinate>
void
StarPolymers<PositionCoordinate, VelocityCoordinate>::performMDTimestep()
{
	const std::size_t particleCount = getParticleCount();

	OPENMPCD_DEBUG_ASSERT(particleCount != 0);

	const std::size_t coordinateCount = 3 * particleCount;
	const FP timestep = getMDTimeStepSize();

	deviceMemoryManager.zeroMemory(d_forces1, coordinateCount);
	deviceMemoryManager.zeroMemory(d_forces2, coordinateCount);

	for(
		std::size_t particleID1 = 0;
		particleID1 < particleCount - 1;
		++particleID1)
	{
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(particleCount - particleID1 - 1)
			ImplementationDetails::StarPolymers::
			updateForceBuffer<VelocityCoordinate>
				<<<gridSize, blockSize>>>(
					particleID1,
					d_positions,
					d_forces1,
					getStarCount(),
					getArmCountPerStar(),
					getParticleCountPerArm(),
					hasMagneticParticles(),
					wcaPotentials,
					fenePotentials,
					magneticPotential,
					workUnitOffset);
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	}
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(3 * particleCount)
		ImplementationDetails::StarPolymers::
		velocityVerletKernel1<<<gridSize, blockSize>>>(
			d_positions,
			d_velocities,
			d_forces1,
			timestep,
			particleCount,
			particleMass,
			workUnitOffset);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	for(
		std::size_t particleID1 = 0;
		particleID1 < particleCount - 1;
		++particleID1)
	{
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(particleCount - particleID1 - 1)
			ImplementationDetails::StarPolymers::
			updateForceBuffer<VelocityCoordinate>
				<<<gridSize, blockSize>>>(
					particleID1,
					d_positions,
					d_forces2,
					getStarCount(),
					getArmCountPerStar(),
					getParticleCountPerArm(),
					hasMagneticParticles(),
					wcaPotentials,
					fenePotentials,
					magneticPotential,
					workUnitOffset);
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	}
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;



	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(3 * particleCount)
		ImplementationDetails::StarPolymers::
		velocityVerletKernel2<<<gridSize, blockSize>>>(
			d_velocities,
			d_forces1,
			d_forces2,
			timestep,
			particleCount,
			particleMass,
			workUnitOffset);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

template<typename PositionCoordinate, typename VelocityCoordinate>
std::size_t
StarPolymers<PositionCoordinate, VelocityCoordinate>::getArmCountPerStar()
const
{
	return armCountPerStar;
}

template<typename PositionCoordinate, typename VelocityCoordinate>
std::size_t
StarPolymers<PositionCoordinate, VelocityCoordinate>::getParticleCountPerArm()
const
{
	return armParticlesPerArm;
}

template<typename PositionCoordinate, typename VelocityCoordinate>
bool
StarPolymers<PositionCoordinate, VelocityCoordinate>::hasMagneticParticles()
const
{
	return hasMagneticParticles_;
}

template<typename PositionCoordinate, typename VelocityCoordinate>
std::size_t
StarPolymers<PositionCoordinate, VelocityCoordinate>::
	getParticleCountPerArmIncludingMagneticParticles()
const
{
	return getParticleCountPerArm() + hasMagneticParticles();
}

template<typename PositionCoordinate, typename VelocityCoordinate>
std::size_t
StarPolymers<PositionCoordinate, VelocityCoordinate>::getParticleCountPerStar()
const
{
	return
		ImplementationDetails::StarPolymers::getParticleCountPerStar(
			getArmCountPerStar(), getParticleCountPerArm(),
			hasMagneticParticles());
}

template<typename PositionCoordinate, typename VelocityCoordinate>
std::size_t
StarPolymers<PositionCoordinate, VelocityCoordinate>::getParticleCount()
const
{
	return
		ImplementationDetails::StarPolymers::getParticleCount(
			getStarCount(), getArmCountPerStar(), getParticleCountPerArm(),
			hasMagneticParticles());
}

template<typename PositionCoordinate, typename VelocityCoordinate>
void
StarPolymers<PositionCoordinate, VelocityCoordinate>::writeStructureToSnapshot(
	VTFSnapshotFile* const snapshot) const
{
	if(snapshot == NULL)
		OPENMPCD_THROW(NULLPointerException, "`snapshot`");

	if(!snapshot->isInWriteMode())
		OPENMPCD_THROW(InvalidArgumentException, "`snapshot` not in write mode");

	if(snapshot->structureBlockHasBeenProcessed())
		OPENMPCD_THROW(InvalidArgumentException, "Cannot write structure block");

	class Helper
	{
	public:
		static
		const std::pair<std::size_t, std::size_t>
		declareAtoms(
			VTFSnapshotFile* const snapshot, const std::size_t count,
			const ParticleType::Enum type)
		{
			const FP radius = -1;
			const std::string name = "";
			std::string typeString;
			switch(type)
			{
				case ParticleType::Core:
					typeString = "Core";
					break;

				case ParticleType::Arm:
					typeString = "Arm";
					break;

				case ParticleType::Magnetic:
					typeString = "Magnetic";
					break;

				default:
					OPENMPCD_THROW(UnimplementedException, "Unknown type");
			}

			return snapshot->declareAtoms(count, radius, name, typeString);
		}
	};

	for(std::size_t star = 0; star < getStarCount(); ++star)
	{
		const std::size_t coreID =
			Helper::declareAtoms(snapshot, 1, ParticleType::Core).first;

		for(std::size_t arm = 0; arm < getArmCountPerStar(); ++arm)
		{
			const std::size_t firstArmParticleID =
			Helper::declareAtoms(
				snapshot, getParticleCountPerArm(), ParticleType::Arm).first;


			for(std::size_t pIA = 0; pIA < getParticleCountPerArm(); ++pIA)
			{
				if(pIA == 0)
				{
					snapshot->declareBond(coreID, firstArmParticleID);
					continue;
				}

				const std::size_t ID = firstArmParticleID + pIA;
				snapshot->declareBond(ID - 1, ID);
			}

			if(hasMagneticParticles())
			{
				const std::size_t ID =
					Helper::declareAtoms(
						snapshot, 1, ParticleType::Magnetic).first;

				if(getParticleCountPerArm() == 0)
				{
					snapshot->declareBond(coreID, ID);
				}
				else
				{
					snapshot->declareBond(ID - 1, ID);
				}
			}
		}
	}
}

template<typename PositionCoordinate, typename VelocityCoordinate>
void
StarPolymers<PositionCoordinate, VelocityCoordinate>::writeStateToSnapshot(
	VTFSnapshotFile* const snapshot) const
{
	if(snapshot == NULL)
		OPENMPCD_THROW(NULLPointerException, "`snapshot`");

	if(snapshot->getNumberOfAtoms() != getParticleCount())
		OPENMPCD_THROW(InvalidArgumentException, "atom count");

	if(!snapshot->isInWriteMode())
		OPENMPCD_THROW(InvalidArgumentException, "`snapshot` not in write mode");

	fetchFromDevice();

	snapshot->writeTimestepBlock(h_positions.get(), h_velocities.get());
}

template<typename PositionCoordinate, typename VelocityCoordinate>
void
StarPolymers<PositionCoordinate, VelocityCoordinate>::readStateFromSnapshot(
	VTFSnapshotFile* const snapshot)
{
	if(snapshot == NULL)
		OPENMPCD_THROW(NULLPointerException, "`snapshot`");

	if(!snapshotStructureIsCompatible(*snapshot))
		OPENMPCD_THROW(InvalidArgumentException, "`snapshot`");

	snapshot->readTimestepBlock(h_positions.get(), h_velocities.get());
	pushToDevice();
}

template<typename PositionCoordinate, typename VelocityCoordinate>
bool
StarPolymers<PositionCoordinate, VelocityCoordinate>::
snapshotStructureIsCompatible(
	const VTFSnapshotFile& snapshot) const
{
	if(!snapshot.isInReadMode())
		return false;

	if(snapshot.getNumberOfAtoms() != getParticleCount())
		return false;

	for(std::size_t p = 0; p < getParticleCount(); ++p)
	{
		const ParticleType::Enum type = getParticleType(p);

		switch(type)
		{
			case ParticleType::Core:
				if(snapshot.getAtomProperties(p).type != std::string("Core"))
					return false;
				break;

			case ParticleType::Arm:
				if(snapshot.getAtomProperties(p).type != std::string("Arm"))
					return false;
				break;

			case ParticleType::Magnetic:
				if(snapshot.getAtomProperties(p).type != std::string("Magnetic"))
					return false;
				break;

			default:
				OPENMPCD_THROW(Exception, "Unexpected");
		}
	}

	for(std::size_t p1 = 0; p1 < getParticleCount(); ++p1)
	{
		for(std::size_t p2 = p1 + 1; p2 < getParticleCount(); ++p2)
		{
			const bool bonded =
				ImplementationDetails::StarPolymers::particlesAreBonded(
					p1, p2,
					getStarCount(),
					getArmCountPerStar(),
					getParticleCountPerArm(),
					hasMagneticParticles());

			if(bonded != snapshot.hasBond(p1, p2))
				return false;
		}
	}

	return true;
}

template<typename PositionCoordinate, typename VelocityCoordinate>
bool
StarPolymers<PositionCoordinate, VelocityCoordinate>::isValidParticleID(
	const std::size_t particleID)
const
{
	return particleID < getParticleCount();
}

template<typename PositionCoordinate, typename VelocityCoordinate>
typename
StarPolymers<PositionCoordinate, VelocityCoordinate>::ParticleType::Enum
StarPolymers<PositionCoordinate, VelocityCoordinate>::getParticleType(
	const std::size_t particleID)
const
{
	#ifdef OPENMPCD_DEBUG
		if(getStarCount() != 1)
			OPENMPCD_THROW(UnimplementedException, "");

		if(!isValidParticleID(particleID))
			OPENMPCD_THROW(OutOfBoundsException, "`particleID`");
	#endif

	return
		ImplementationDetails::StarPolymers::getParticleType(
				particleID,
				getStarCount(),
				getArmCountPerStar(),
				getParticleCountPerArm(),
				hasMagneticParticles());
}

template<typename PositionCoordinate, typename VelocityCoordinate>
std::size_t
StarPolymers<PositionCoordinate, VelocityCoordinate>::getParticleID(
	const std::size_t star, const std::size_t arm,
	const std::size_t particleInArm)
const
{
	#ifdef OPENMPCD_DEBUG
		if(star >= getStarCount())
			OPENMPCD_THROW(OutOfBoundsException, "`star`");
		if(arm >= getArmCountPerStar())
			OPENMPCD_THROW(OutOfBoundsException, "`arm`");
		if(particleInArm >= getParticleCountPerArm())
			OPENMPCD_THROW(OutOfBoundsException, "`particleInArm`");
	#endif

	return
		star * getParticleCountPerStar()
		+ 1 //core particle
		+ arm * getParticleCountPerArmIncludingMagneticParticles()
		+ particleInArm;
}

template<typename PositionCoordinate, typename VelocityCoordinate>
std::size_t
StarPolymers<PositionCoordinate, VelocityCoordinate>::getParticleID(
	const std::size_t star, const std::size_t arm)
const
{
	#ifdef OPENMPCD_DEBUG
		if(star >= getStarCount())
			OPENMPCD_THROW(OutOfBoundsException, "`star`");
		if(arm >= getArmCountPerStar())
			OPENMPCD_THROW(OutOfBoundsException, "`arm`");
	#endif

	return
		star * getParticleCountPerStar()
		+ 1 //core particle
		+ arm * getParticleCountPerArmIncludingMagneticParticles()
		+ getParticleCountPerArm();
}

template<typename PositionCoordinate, typename VelocityCoordinate>
std::size_t
StarPolymers<PositionCoordinate, VelocityCoordinate>::getParticleID(
	const std::size_t star)
const
{
	#ifdef OPENMPCD_DEBUG
		if(star >= getStarCount())
			OPENMPCD_THROW(OutOfBoundsException, "`star`");
	#endif

	return star * getParticleCountPerStar();
}

template<typename PositionCoordinate, typename VelocityCoordinate>
void
StarPolymers<PositionCoordinate, VelocityCoordinate>::
	initializePositionsOnHost()
{
	#ifdef OPENMPCD_DEBUG
		if(getStarCount() != 1)
			OPENMPCD_THROW(UnimplementedException, "");
	#endif

	static const PositionCoordinate beadToBeadDistance = 1;

	const std::size_t star = 0;
	for(std::size_t arm = 0; arm < getArmCountPerStar(); ++arm)
	{
		const std::size_t coreID = getParticleID(star);

		h_positions[3 * coreID + 0] = 0;
		h_positions[3 * coreID + 1] = 0;
		h_positions[3 * coreID + 2] = 0;

		for(std::size_t pIA = 0; pIA < getParticleCountPerArm(); ++pIA)
		{
			const std::size_t particleID = getParticleID(star, arm, pIA);

			const PositionCoordinate angle =
				2 * boost::math::constants::pi<PositionCoordinate>()
				* arm / getArmCountPerStar();

			const PositionCoordinate distance =
				( pIA + 1 ) * beadToBeadDistance;

			h_positions[3 * particleID + 0] = cos(angle) * distance;
			h_positions[3 * particleID + 1] = sin(angle) * distance;
			h_positions[3 * particleID + 2] = 0;
		}

		if(hasMagneticParticles())
		{
			const std::size_t particleID = getParticleID(star, arm);

			const PositionCoordinate angle =
				2 * boost::math::constants::pi<PositionCoordinate>()
				* arm / getArmCountPerStar();

			const PositionCoordinate distance =
				( getParticleCountPerArm() + 1 ) * beadToBeadDistance;

			h_positions[3 * particleID + 0] = cos(angle) * distance;
			h_positions[3 * particleID + 1] = sin(angle) * distance;
			h_positions[3 * particleID + 2] = 0;
		}
	}
}

template<typename PositionCoordinate, typename VelocityCoordinate>
void
StarPolymers<PositionCoordinate, VelocityCoordinate>::
	initializeVelocitiesOnHost()
{
	for(std::size_t star = 0; star < getStarCount(); ++star)
	{
		const std::size_t coreID = getParticleID(star);

		h_velocities[3 * coreID + 0] = 0;
		h_velocities[3 * coreID + 1] = 0;
		h_velocities[3 * coreID + 2] = 0;

		for(std::size_t arm = 0; arm < getArmCountPerStar(); ++arm)
		{
			for(std::size_t pIA = 0; pIA < getParticleCountPerArm(); ++pIA)
			{
				const std::size_t particleID = getParticleID(star, arm, pIA);

				h_velocities[3 * particleID + 0] = 0;
				h_velocities[3 * particleID + 1] = 0;
				h_velocities[3 * particleID + 2] = 0;
			}

			if(hasMagneticParticles())
			{
				const std::size_t particleID = getParticleID(star, arm);

				h_velocities[3 * particleID + 0] = 0;
				h_velocities[3 * particleID + 1] = 0;
				h_velocities[3 * particleID + 2] = 0;
			}
		}
	}
}

} //namespace MPCSolute
} //namespace CUDA
} //namespace OpenMPCD

template class OpenMPCD::CUDA::MPCSolute::StarPolymers<double, double>;
//Type `float` is not currently tested in the unit tests, since it
//seems to result in considerable inaccuracies with the magnetic interaction.
//Adapt tests before instantiation with `float` here.
