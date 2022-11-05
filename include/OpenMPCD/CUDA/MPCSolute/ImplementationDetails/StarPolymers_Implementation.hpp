/**
 * @file
 * Contains implementation details used in
 * `OpenMPCD::CUDA::MPCSolute::StarPolymers`.
 */

#ifndef OPENMPCD_CUDA_MPCSOLUTE_IMPLEMENTATIONDETAILS_STARPOLYMERS_IMPLEMENTATION_HPP
#define OPENMPCD_CUDA_MPCSOLUTE_IMPLEMENTATIONDETAILS_STARPOLYMERS_IMPLEMENTATION_HPP

#include <OpenMPCD/CUDA/MPCSolute/StarPolymers.hpp>

#include <OpenMPCD/CUDA/Macros.hpp>

#include <algorithm>

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

inline
OPENMPCD_CUDA_HOST_AND_DEVICE
std::size_t getParticleCountPerStar(
	const std::size_t armCountPerStar,
	const std::size_t particleCountPerArm,
	const bool hasMagneticParticles)
{
	return
		1 //core
		+
		armCountPerStar * (particleCountPerArm + hasMagneticParticles);
}

inline
OPENMPCD_CUDA_HOST_AND_DEVICE
std::size_t getParticleCount(
	const std::size_t starCount,
	const std::size_t armCountPerStar,
	const std::size_t particleCountPerArm,
	const bool hasMagneticParticles)
{
	const std::size_t particleCountPerStar = getParticleCountPerStar(
		armCountPerStar, particleCountPerArm, hasMagneticParticles);

	return starCount * particleCountPerStar;
}

inline
OPENMPCD_CUDA_HOST_AND_DEVICE
void getParticleStructureIndices(
	const std::size_t particleID,
	const std::size_t starCount,
	const std::size_t armCountPerStar,
	const std::size_t particleCountPerArm,
	const bool hasMagneticParticles,
	std::size_t* const starID,
	bool* const isCoreParticle,
	std::size_t* const armID,
	bool* const isMagneticParticle,
	std::size_t* const particleIDInArm)
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		starID != NULL, OpenMPCD::NULLPointerException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		isCoreParticle != NULL, OpenMPCD::NULLPointerException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		armID != NULL, OpenMPCD::NULLPointerException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		isMagneticParticle != NULL, OpenMPCD::NULLPointerException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		particleIDInArm != NULL, OpenMPCD::NULLPointerException);

	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		particleID <
			getParticleCount(
				starCount, armCountPerStar, particleCountPerArm,
				hasMagneticParticles),
		OpenMPCD::InvalidArgumentException);

	const std::size_t particleCountPerStar =
		getParticleCountPerStar(
			armCountPerStar, particleCountPerArm, hasMagneticParticles);

	*starID = particleID / particleCountPerStar;
	const std::size_t particleIDWithinStar = particleID % particleCountPerStar;

	if(particleIDWithinStar == 0)
	{
		*isCoreParticle = true;
		*isMagneticParticle = false;
		return;
	}

	*isCoreParticle = false;

	const std::size_t particleCountPerExtendedArm =
		particleCountPerArm + hasMagneticParticles;

	*armID = (particleIDWithinStar - 1) / particleCountPerExtendedArm;
	*particleIDInArm = (particleIDWithinStar - 1) % particleCountPerExtendedArm;
	*isMagneticParticle = *particleIDInArm == particleCountPerArm;
}

inline
OPENMPCD_CUDA_HOST_AND_DEVICE
ParticleType::Enum getParticleType(
	const std::size_t particleID,
	const std::size_t starCount,
	const std::size_t armCountPerStar,
	const std::size_t particleCountPerArm,
	const bool hasMagneticParticles)
{
	std::size_t starID, armID, particleIDInArm;
	bool isCoreParticle, isMagneticParticle;

	getParticleStructureIndices(
		particleID,
		starCount, armCountPerStar,	particleCountPerArm, hasMagneticParticles,
		&starID, &isCoreParticle, &armID, &isMagneticParticle,
		&particleIDInArm);

	if(isCoreParticle)
		return ParticleType::Core;

	if(isMagneticParticle)
		return ParticleType::Magnetic;

	return ParticleType::Arm;
}

inline
OPENMPCD_CUDA_HOST_AND_DEVICE
bool particlesAreBonded(
	const std::size_t particleID1,
	const std::size_t particleID2,
	const std::size_t starCount,
	const std::size_t armCountPerStar,
	const std::size_t particleCountPerArm,
	const bool hasMagneticParticles)
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		particleID1 != particleID2,
		OpenMPCD::InvalidArgumentException);

	std::size_t starID1, armID1, particleIDInArm1;
	bool isCoreParticle1, isMagneticParticle1;

	std::size_t starID2, armID2, particleIDInArm2;
	bool isCoreParticle2, isMagneticParticle2;

	getParticleStructureIndices(
		particleID1,
		starCount, armCountPerStar,	particleCountPerArm, hasMagneticParticles,
		&starID1, &isCoreParticle1, &armID1, &isMagneticParticle1,
		&particleIDInArm1);

	getParticleStructureIndices(
		particleID2,
		starCount, armCountPerStar,	particleCountPerArm, hasMagneticParticles,
		&starID2, &isCoreParticle2, &armID2, &isMagneticParticle2,
		&particleIDInArm2);

	if(starID1 != starID2)
		return false;

	#ifdef OPENMPCD_COMPILER_GCC
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
	#endif
	if(isCoreParticle1)
		return particleIDInArm2 == 0;
	if(isCoreParticle2)
		return particleIDInArm1 == 0;
	#ifdef OPENMPCD_COMPILER_GCC
		#pragma GCC diagnostic pop
	#endif

	if(armID1 != armID2)
		return false;

	if(particleIDInArm1 + 1 == particleIDInArm2)
		return true;
	if(particleIDInArm2 + 1 == particleIDInArm1)
		return true;

	return false;
}

inline
OPENMPCD_CUDA_HOST_AND_DEVICE
std::size_t getParticleTypeCombinationIndex(
	const ParticleType::Enum type1, const ParticleType::Enum type2)
{
	#ifndef __CUDA_ARCH__
		using std::min;
		using std::max;
	#endif

	const std::size_t lower =
		min(static_cast<std::size_t>(type1), static_cast<std::size_t>(type2));
	const std::size_t upper =
		max(static_cast<std::size_t>(type1), static_cast<std::size_t>(type2));

	//WARNING: this won't work anymore if there are more than three types!
	const std::size_t tmp = (upper << 1) + lower;
	if(tmp == 0)
		return 0;
	return tmp - 1;
}


/**
 * Constructs the interaction potentials in Device memory.
 *
 * The interaction parameters are described in the documentation of
 * `OpenMPCD::CUDA::MPCSolute::StarPolymers`.
 *
 * The arrays returned in the output parameters `WCAPotentials` and
 * `FENEPotentials` contain the interactions of particles of type `type1` and
 * `type2` at the array index `getParticleTypeCombinationIndex(type1, type2)`.
 *
 * @tparam T The underlying data type.
 *
 * @param[in]  epsilon_core      The \f$ \varepsilon_C \f$ parameter.
 * @param[in]  epsilon_arm       The \f$ \varepsilon_A \f$ parameter.
 * @param[in]  epsilon_magnetic  The \f$ \varepsilon_M \f$ parameter.
 * @param[in]  sigma_core        The \f$ \sigma_C \f$ parameter.
 * @param[in]  sigma_arm         The \f$ \sigma_A \f$ parameter.
 * @param[in]  sigma_magnetic    The \f$ \sigma_M \f$ parameter.
 * @param[in]  D_core            The \f$ D_C \f$ parameter.
 * @param[in]  D_arm             The \f$ D_A \f$ parameter.
 * @param[in]  D_magnetic        The \f$ D_M \f$ parameter.
 * @param[in]  magneticPrefactor The `prefactor` argument to the magnetic
 *                               interaction.
 * @param[in]  dipoleOrientation Normalized vector that defines the orientation
 *                               of the magnetic dipoles.
 * @param[in] WCAPotentials      Where to construct the WCA potential instances.
 * @param[in] FENEPotentials     Where to construct the FENE potential
 *                               instances.
 * @param[in] magneticPotential  Where to construct the magnetic potential
 *                               instances.
 */
template<typename T>
__global__
void createInteractionsOnDevice_kernel(
	const T epsilon_core, const T epsilon_arm, const T epsilon_magnetic,
	const T sigma_core, const T sigma_arm, const T sigma_magnetic,
	const T D_core, const T D_arm, const T D_magnetic,
	const T magneticPrefactor, const Vector3D<T> dipoleOrientation,
	PairPotentials::WeeksChandlerAndersen_DistanceOffset<T>** const
		WCAPotentials,
	PairPotentials::FENE<T>** const FENEPotentials,
	PairPotentials::
		MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<T>** const
			magneticPotential
	)
{
	OPENMPCD_DEBUG_ASSERT(WCAPotentials != NULL);
	OPENMPCD_DEBUG_ASSERT(FENEPotentials != NULL);
	OPENMPCD_DEBUG_ASSERT(magneticPotential != NULL);

	typedef PairPotentials::WeeksChandlerAndersen_DistanceOffset<T> WCA;
	typedef PairPotentials::FENE<T> FENE;
	typedef
		PairPotentials::
			MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<T>
		Magnetic;

	const std::size_t indexCC =
		getParticleTypeCombinationIndex(
			ParticleType::Core, ParticleType::Core);
	const std::size_t indexCA =
		getParticleTypeCombinationIndex(
			ParticleType::Core, ParticleType::Arm);
	const std::size_t indexCM =
		getParticleTypeCombinationIndex(
			ParticleType::Core, ParticleType::Magnetic);
	const std::size_t indexAA =
		getParticleTypeCombinationIndex(
			ParticleType::Arm, ParticleType::Arm);
	const std::size_t indexAM =
		getParticleTypeCombinationIndex(
			ParticleType::Arm, ParticleType::Magnetic);
	const std::size_t indexMM =
		getParticleTypeCombinationIndex(
			ParticleType::Magnetic, ParticleType::Magnetic);


	const T epsilon_CC = epsilon_core;
	const T epsilon_CA = sqrt(epsilon_core * epsilon_arm);
	const T epsilon_CM = sqrt(epsilon_core * epsilon_magnetic);
	const T epsilon_AA = epsilon_arm;
	const T epsilon_AM = sqrt(epsilon_arm * epsilon_magnetic);
	const T epsilon_MM = epsilon_magnetic;

	const T sigma_CC = sigma_core;
	const T sigma_CA = 0.5 * (sigma_core + sigma_arm);
	const T sigma_CM = 0.5 * (sigma_core + sigma_magnetic);
	const T sigma_AA = sigma_arm;
	const T sigma_AM = 0.5 * (sigma_arm + sigma_magnetic);
	const T sigma_MM = sigma_magnetic;

	const T D_CC = D_core;
	const T D_CA = 0.5 * (D_core + D_arm);
	const T D_CM = 0.5 * (D_core + D_magnetic);
	const T D_AA = D_arm;
	const T D_AM = 0.5 * (D_arm + D_magnetic);
	const T D_MM = D_magnetic;

	WCAPotentials[indexCC] = new WCA(epsilon_CC, sigma_CC, D_CC);
	WCAPotentials[indexCA] = new WCA(epsilon_CA, sigma_CA, D_CA);
	WCAPotentials[indexCM] = new WCA(epsilon_CM, sigma_CM, D_CM);
	WCAPotentials[indexAA] = new WCA(epsilon_AA, sigma_AA, D_AA);
	WCAPotentials[indexAM] = new WCA(epsilon_AM, sigma_AM, D_AM);
	WCAPotentials[indexMM] = new WCA(epsilon_MM, sigma_MM, D_MM);

	const T l_0_CA = D_CA;
	const T l_0_AA = D_AA;
	const T l_0_AM = D_AM;

	const T R_CA = 1.5 * sigma_CA;
	const T R_AA = 1.5 * sigma_AA;
	const T R_AM = 1.5 * sigma_AM;

	const T K_CA = 30 * epsilon_CA / (sigma_CA * sigma_CA);
	const T K_AA = 30 * epsilon_AA / (sigma_AA * sigma_AA);
	const T K_AM = 30 * epsilon_AM / (sigma_AM * sigma_AM);

	FENEPotentials[indexCA] = new FENE(K_CA, l_0_CA, R_CA);
	FENEPotentials[indexAA] = new FENE(K_AA, l_0_AA, R_AA);
	FENEPotentials[indexAM] = new FENE(K_AM, l_0_AM, R_AM);

	#ifdef OPENMPCD_DEBUG
		FENEPotentials[indexCC] = NULL;
		FENEPotentials[indexCM] = NULL;
		FENEPotentials[indexMM] = NULL;
	#endif

	*magneticPotential = new Magnetic(magneticPrefactor, dipoleOrientation);
}

template<typename T>
void createInteractionsOnDevice(
	const T epsilon_core, const T epsilon_arm, const T epsilon_magnetic,
	const T sigma_core, const T sigma_arm, const T sigma_magnetic,
	const T D_core, const T D_arm, const T D_magnetic,
	const T magneticPrefactor, const Vector3D<T> dipoleOrientation,
	PairPotentials::WeeksChandlerAndersen_DistanceOffset<T>*** const
		WCAPotentials,
	PairPotentials::FENE<T>*** const FENEPotentials,
	PairPotentials::
		MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<T>*** const
			magneticPotential
	)
{
	OPENMPCD_DEBUG_ASSERT(WCAPotentials != NULL);
	OPENMPCD_DEBUG_ASSERT(FENEPotentials != NULL);
	OPENMPCD_DEBUG_ASSERT(magneticPotential != NULL);

	DeviceMemoryManager::allocateMemoryUnregistered(WCAPotentials, 6);
	DeviceMemoryManager::allocateMemoryUnregistered(FENEPotentials, 6);
	DeviceMemoryManager::allocateMemoryUnregistered(magneticPotential, 1);

	createInteractionsOnDevice_kernel<<<1, 1>>>(
		epsilon_core, epsilon_arm, epsilon_magnetic,
		sigma_core, sigma_arm, sigma_magnetic,
		D_core, D_arm, D_magnetic,
		magneticPrefactor, dipoleOrientation,
		*WCAPotentials, *FENEPotentials, *magneticPotential
		);
}


/**
 * Frees the memory allocated through `createInteractionsOnDevice_kernel`.
 *
 * @param[in] WCAPotentials     The array of allocated WCA potentials on the
 *                              CUDA Device.
 * @param[in] FENEPotentials    The array of allocated FENE potentials on the
 *                              CUDA Device.
 * @param[in] magneticPotential The allocated magnetic potential on the CUDA
 *                              Device.
 */
template<typename T>
__global__
void destroyInteractionsOnDevice_kernel(
	PairPotentials::WeeksChandlerAndersen_DistanceOffset<T>** const
		WCAPotentials,
	PairPotentials::FENE<T>** const FENEPotentials,
	PairPotentials::
		MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<T>** const
			magneticPotential
	)
{
	OPENMPCD_DEBUG_ASSERT(WCAPotentials != NULL);
	OPENMPCD_DEBUG_ASSERT(FENEPotentials != NULL);
	OPENMPCD_DEBUG_ASSERT(magneticPotential != NULL);

	for(std::size_t i = 0; i < 6; ++i)
	{
		OPENMPCD_DEBUG_ASSERT(WCAPotentials[i] != NULL);

		delete WCAPotentials[i];
	}

	const std::size_t indexCA =
		getParticleTypeCombinationIndex(
			ParticleType::Core, ParticleType::Arm);
	const std::size_t indexAA =
		getParticleTypeCombinationIndex(
			ParticleType::Arm, ParticleType::Arm);
	const std::size_t indexAM =
		getParticleTypeCombinationIndex(
			ParticleType::Arm, ParticleType::Magnetic);

	OPENMPCD_DEBUG_ASSERT(FENEPotentials[indexCA] != NULL);
	OPENMPCD_DEBUG_ASSERT(FENEPotentials[indexAA] != NULL);
	OPENMPCD_DEBUG_ASSERT(FENEPotentials[indexAM] != NULL);

	#ifdef OPENMPCD_DEBUG
		const std::size_t indexCC =
			getParticleTypeCombinationIndex(
				ParticleType::Core, ParticleType::Core);
		const std::size_t indexCM =
			getParticleTypeCombinationIndex(
				ParticleType::Core, ParticleType::Magnetic);
		const std::size_t indexMM =
			getParticleTypeCombinationIndex(
				ParticleType::Magnetic, ParticleType::Magnetic);

		OPENMPCD_DEBUG_ASSERT(FENEPotentials[indexCC] == NULL);
		OPENMPCD_DEBUG_ASSERT(FENEPotentials[indexCM] == NULL);
		OPENMPCD_DEBUG_ASSERT(FENEPotentials[indexMM] == NULL);
	#endif

	delete FENEPotentials[indexCA];
	delete FENEPotentials[indexAA];
	delete FENEPotentials[indexAM];

	delete *magneticPotential;
}

template<typename T>
void destroyInteractionsOnDevice(
	PairPotentials::WeeksChandlerAndersen_DistanceOffset<T>** const
		WCAPotentials,
	PairPotentials::FENE<T>** const FENEPotentials,
	PairPotentials::
		MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<T>** const
			magneticPotential
	)
{
	destroyInteractionsOnDevice_kernel<<<1, 1>>>(
		WCAPotentials, FENEPotentials, magneticPotential);

	DeviceMemoryManager::freeMemoryUnregistered(WCAPotentials);
	DeviceMemoryManager::freeMemoryUnregistered(FENEPotentials);
	DeviceMemoryManager::freeMemoryUnregistered(magneticPotential);
}


template<typename T>
OPENMPCD_CUDA_HOST_AND_DEVICE
const Vector3D<T>
computeForceOnParticle1DueToParticle2(
	const std::size_t particleID1,
	const std::size_t particleID2,
	const RemotelyStoredVector<const T>& position1,
	const RemotelyStoredVector<const T>& position2,
	const std::size_t starCount,
	const std::size_t armCountPerStar,
	const std::size_t particleCountPerArm,
	const bool hasMagneticParticles,
	PairPotentials::WeeksChandlerAndersen_DistanceOffset<T>** const
		WCAPotentials,
	PairPotentials::FENE<T>** const FENEPotentials,
	PairPotentials::
		MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<T>** const
			magneticPotential)
{
	OPENMPCD_DEBUG_ASSERT(particleID1 != particleID2);

	const ParticleType::Enum particleType1 =
		getParticleType(
			particleID1,
			starCount, armCountPerStar, particleCountPerArm,
			hasMagneticParticles);

	const ParticleType::Enum particleType2 =
		getParticleType(
			particleID2,
			starCount, armCountPerStar, particleCountPerArm,
			hasMagneticParticles);

	const std::size_t particleTypeCombinationIndex =
		getParticleTypeCombinationIndex(particleType1, particleType2);

	Vector3D<T> forceOnParticle1 =
		WCAPotentials[particleTypeCombinationIndex]->
			forceOnR1DueToR2(position1, position2);


	if(
		particleType1 == ParticleType::Magnetic &&
		particleType2 == ParticleType::Magnetic
	  )
	{
		forceOnParticle1 +=
			magneticPotential[0]->forceOnR1DueToR2(position1, position2);
		return forceOnParticle1;
	}

	const bool bonded =
		particlesAreBonded(
			particleID1, particleID2,
			starCount, armCountPerStar, particleCountPerArm,
			hasMagneticParticles);

	if(!bonded)
		return forceOnParticle1;

	forceOnParticle1 +=
		FENEPotentials[particleTypeCombinationIndex]->
			forceOnR1DueToR2(position1, position2);

	return forceOnParticle1;
}

} //namespace StarPolymers
} //namespace ImplementationDetails
} //namespace MPCSolute
} //namespace CUDA
} //namespace OpenMPCD

#endif //OPENMPCD_CUDA_MPCSOLUTE_IMPLEMENTATIONDETAILS_STARPOLYMERS_IMPLEMENTATION_HPP
