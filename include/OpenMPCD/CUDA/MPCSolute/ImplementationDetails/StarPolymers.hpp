/**
 * @file
 * Contains implementation details used in
 * `OpenMPCD::CUDA::MPCSolute::StarPolymers`.
 */

#ifndef OPENMPCD_CUDA_MPCSOLUTE_IMPLEMENTATIONDETAILS_STARPOLYMERS_HPP
#define OPENMPCD_CUDA_MPCSOLUTE_IMPLEMENTATIONDETAILS_STARPOLYMERS_HPP

#include <OpenMPCD/CUDA/MPCSolute/StarPolymers.hpp>

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCSolute
{

/**
 * Contains implementation details used in `OpenMPCD::CUDA::MPCSolute`.
 */
namespace ImplementationDetails
{

/**
 * Contains implementation details used in
 * `OpenMPCD::CUDA::MPCSolute::StarPolymers`.
 */
namespace StarPolymers
{

/**
 * Returns the total number of particles per star.
 *
 * @param[in]  armCountPerStar
 *             The number of polymer arms per star.
 * @param[in]  particleCountPerArm
 *             The number of non-magnetic particles per arm.
 * @param[in]  hasMagneticParticles
 *             Whether each arm has an additional, magnetic particle attached.
 */
OPENMPCD_CUDA_HOST_AND_DEVICE
std::size_t getParticleCountPerStar(
	const std::size_t armCountPerStar,
	const std::size_t particleCountPerArm,
	const bool hasMagneticParticles);

/**
 * Returns the total number of particles in a `StarPolymers` instance.
 *
 * @param[in]  starCount
 *             The number of stars in the `StarPolymers` instance.
 * @param[in]  armCountPerStar
 *             The number of polymer arms per star.
 * @param[in]  particleCountPerArm
 *             The number of non-magnetic particles per arm.
 * @param[in]  hasMagneticParticles
 *             Whether each arm has an additional, magnetic particle attached.
 */
OPENMPCD_CUDA_HOST_AND_DEVICE
std::size_t getParticleCount(
	const std::size_t starCount,
	const std::size_t armCountPerStar,
	const std::size_t particleCountPerArm,
	const bool hasMagneticParticles);

/**
 * Computes the structure indices of the particle ID given.
 *
 * The pointers given must not correspond to overlapping regions of memory.
 *
 * @throw OpenMPCD::NULLPointerException
 *        If `OPENMPCD_DEBUG` is defined, throws if either `starID`,
 *        `isCoreParticle`, `armID`, `isMagneticParticl`, or `particleIDInArm`
 *        is `nullptr`.
 *
 * @throw OpenMPCD::InvalidArgumentException
 *        If `OPENMPCD_DEBUG` is defined, throws if `particleID` is greater than
 *        or equal to the number of particles, as returned by
 *        `getParticleCount`.
 *
 * @param[in]  particleID
 *             The particle ID to query.
 * @param[in]  starCount
 *             The number of stars in the `StarPolymers` instance.
 * @param[in]  armCountPerStar
 *             The number of polymer arms per star.
 * @param[in]  particleCountPerArm
 *             The number of non-magnetic particles per arm.
 * @param[in]  hasMagneticParticles
 *             Whether each arm has an additional, magnetic particle attached.
 * @param[out] starID
 *             The ID of the star the particle is in, running from `0` to
 *             `starCount - 1`, inclusive.
 * @param[out] isCoreParticle
 *             Whether the `particleID` corresponds to the star's core particle.
 * @param[out] armID
 *             If the particle is not a core particle, this is set the
 *             particle's arm's ID within the star, running from `0` to
 *             `armCountPerStar - 1`, inclusive. Otherwise, the value is left
 *             unchanged.
 * @param[out] isMagneticParticle
 *             Whether the `particleID` corresponds to a magnetic particle.
 * @param[out] particleIDInArm
 *             If `isCoreParticle`, the value is left unchanged. Otherwise, if
 *             the `particleID` corresponds to a magnetic particle, this is set
 *             to `particleCountPerArm`. Otherwise, the `particleID` corresponds
 *             to an arm particle, and this value is set to the particle's ID
 *             within the arm, running from `0` to `particleCountPerArm - 1`,
 *             inclusive.
 */
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
	std::size_t* const particleIDInArm);

/**
 * Returns the particle type that corresponds to the given ID.
 *
 * @throw OpenMPCD::InvalidArgumentException
 *        If `OPENMPCD_DEBUG` is defined, throws if `particleID` is greater than
 *        or equal to the number of particles, as returned by
 *        `getParticleCount`.
 *
 * @param[in]  particleID
 *             The particle ID to query.
 * @param[in]  starCount
 *             The number of stars in the `StarPolymers` instance.
 * @param[in]  armCountPerStar
 *             The number of polymer arms per star.
 * @param[in]  particleCountPerArm
 *             The number of non-magnetic particles per arm.
 * @param[in]  hasMagneticParticles
 *             Whether each arm has an additional, magnetic particle attached.
 */
OPENMPCD_CUDA_HOST_AND_DEVICE
ParticleType::Enum getParticleType(
	const std::size_t particleID,
	const std::size_t starCount,
	const std::size_t armCountPerStar,
	const std::size_t particleCountPerArm,
	const bool hasMagneticParticles);

/**
 * Returns whether the two given particles are bonded.
 *
 * @throw OpenMPCD::InvalidArgumentException
 *        If `OPENMPCD_DEBUG` is defined, throws if either `particleID1` or
 *        `particleID2` is greater than or equal to the number of particles, as
 *        returned by `getParticleCount`, or if `particleID1 == particleID2`.
 *
 * @param[in]  particleID1
 *             The ID of the first particle.
 * @param[in]  particleID2
 *             The ID of the second particle, which must not be equal to
 *             `particleID1`.
 * @param[in]  starCount
 *             The number of stars in the `StarPolymers` instance.
 * @param[in]  armCountPerStar
 *             The number of polymer arms per star.
 * @param[in]  particleCountPerArm
 *             The number of non-magnetic particles per arm.
 * @param[in]  hasMagneticParticles
 *             Whether each arm has an additional, magnetic particle attached.
 */
OPENMPCD_CUDA_HOST_AND_DEVICE
bool particlesAreBonded(
	const std::size_t particleID1,
	const std::size_t particleID2,
	const std::size_t starCount,
	const std::size_t armCountPerStar,
	const std::size_t particleCountPerArm,
	const bool hasMagneticParticles);

/**
 * Returns a different integer, in a consecutive range starting from `0`, for
 * every combination of particle types, except that the result is invariant
 * under exchange of the two arguments.
 *
 * @param[in] type1 The particle type of the first particle.
 * @param[in] type2 The particle type of the second particle.
 */
OPENMPCD_CUDA_HOST_AND_DEVICE
std::size_t getParticleTypeCombinationIndex(
	const ParticleType::Enum type1, const ParticleType::Enum type2);

/**
 * Constructs the necessary interaction potentials in memory.
 *
 * The interaction parameters are described in the documentation of
 * `OpenMPCD::CUDA::MPCSolute::StarPolymers`.
 *
 * The arrays returned in the output parameters `WCAPotentials` and
 * `FENEPotentials` contain the interactions of particles of type `type1` and
 * `type2` at the array index `getParticleTypeCombinationIndex(type1, type2)`.
 *
 * @throw OpenMPCD::NULLPointerException
 *        If `OPENMPCD_DEBUG` is defined, throws if either of the output
 *        arguments is `nullptr`.
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
 * @param[out] WCAPotentials     Returns an array of WCA potential instance
 *                               pointers on the CUDA Device.
 *                               The memory allocated this way has to be freed
 *                               by `destroyInteractionOnDevice`.
 * @param[out] FENEPotentials    Returns an array of FENE potential instance
 *                               pointers on the CUDA Device.
 *                               The memory allocated this way has to be freed
 *                               by `destroyInteractionOnDevice`.
 * @param[out] magneticPotential Returns a pointer to an instance pointer for
 *                               the magnetic potential on the CUDA Device.
 *                               The memory allocated this way has to be freed
 *                               by `destroyInteractionOnDevice`.
 */
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
	);

/**
 * Frees the memory allocated through `createInteractionsOnDevice`.
 *
 * @throw OpenMPCD::NULLPointerException
 *        If `OPENMPCD_DEBUG` is defined, throws if either of the arguments is
 *        `nullptr`, or if `WCAPotentials` or `FENEPotentials` are not arrays
 *        with non-`nullptr` entries at the indices returned by
 *        `getParticleTypeCombinationIndex` for the appropriate type
 *        combinations (all possible combinations for `WCAPotentials`, and
 *        Core-Arm, Arm-Arm, or Arm-Magnetic combinations for `FENEPotentials`),
 *        or if the entries are non-`nullptr` for invalid type combination
 *        indices.
 *
 * @param[in] WCAPotentials     The array of allocated WCA potentials on the
 *                              CUDA Device.
 * @param[in] FENEPotentials    The array of allocated FENE potentials on the
 *                              CUDA Device.
 * @param[in] magneticPotential The allocated magnetic potential on the CUDA
 *                              Device.
 */
template<typename T>
void destroyInteractionsOnDevice(
	PairPotentials::WeeksChandlerAndersen_DistanceOffset<T>** const
		WCAPotentials,
	PairPotentials::FENE<T>** const FENEPotentials,
	PairPotentials::
		MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<T>** const
			magneticPotential
	);

/**
 * Returns the force that is exerted on `particleID1` due to `particleID2`.
 *
 * The interactions passed in this function are additive, i.e. if the two
 * particles match the criteria of multiple interactions, these matching
 * interactions will all be applied.
 *
 * @tparam T The underlying numeric type.
 *
 * @param[in] particleID1
 *            The ID of the first particle.
 * @param[in] particleID2
 *            The ID of the second particle, which must not be equal to
 *            `particleID1`.
 * @param[in] position1
 *            The position of the first particle.
 * @param[in] position2
 *            The position of the second particle.
 * @param[in] starCount
 *            The number of stars in the `StarPolymers` instance.
 * @param[in] armCountPerStar
 *            The number of polymer arms per star.
 * @param[in] particleCountPerArm
 *            The number of non-magnetic particles per arm.
 * @param[in] hasMagneticParticles
 *            Whether each arm has an additional, magnetic particle attached.
 * @param[in] WCAPotentials
 *            The array of allocated WCA potentials.
 * @param[in] FENEPotentials
 *            The array of allocated FENE potentials.
 * @param[in] magneticPotential
 *            The allocated magnetic potential.
 */
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
			magneticPotential);

} //namespace StarPolymers
} //namespace ImplementationDetails
} //namespace MPCSolute
} //namespace CUDA
} //namespace OpenMPCD

#include <OpenMPCD/CUDA/MPCSolute/ImplementationDetails/StarPolymers_Implementation.hpp>

#endif //OPENMPCD_CUDA_MPCSOLUTE_IMPLEMENTATIONDETAILS_STARPOLYMERS_HPP
