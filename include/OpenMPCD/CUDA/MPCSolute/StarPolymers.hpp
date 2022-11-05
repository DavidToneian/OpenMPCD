/**
 * @file
 * Defines the `OpenMPCD::CUDA::MPCSolute::StarPolymers` class.
 */

#ifndef OPENMPCD_CUDA_MPCSOLUTE_STARPOLYMERS_HPP
#define OPENMPCD_CUDA_MPCSOLUTE_STARPOLYMERS_HPP

#include <OpenMPCD/CUDA/MPCSolute/Base.hpp>

#include <OpenMPCD/CUDA/MPCSolute/ImplementationDetails/StarPolymers_ParticleType.hpp>

#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/CUDA/BoundaryCondition/LeesEdwards.hpp>
#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/PairPotentials/FENE.hpp>
#include <OpenMPCD/PairPotentials/MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles.hpp>
#include <OpenMPCD/PairPotentials/WeeksChandlerAndersen_DistanceOffset.hpp>
#include <OpenMPCD/VTFSnapshotFile.hpp>

#include <boost/scoped_array.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCSolute
{

/**
 * Class representing star polymers.
 *
 * A star polymer consists of one central core particle, attached to which there
 * are a number of linear, homogeneous polymer chains, called "arms". The ends
 * of the arms may be functionalized, by attaching an additional particle that
 * carries a magnetic dipole moment.
 *
 * The following three interaction potentials (magnetic, FENE, and WCA) are
 * cumulative, i.e. a pair of particles can be subject to interaction due to two
 * or more interactions at the same time.
 *
 * The functionalized magnetic particles interact with one another via the
 * `OpenMPCD::PairPotentials::MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles`
 * interaction.
 * The `prefactor` property of the interaction class is set according to the
 * `interactionParameters.magneticPrefactor` configuration property.
 * The orientation of the dipoles may be set via the
 * `interactionParameters.dipoleOrientation` setting, which must be an array of
 * three floating-point values which define an axis. The vector formed by these
 * three values need not be normalized; its magnitude is of no relevance.
 *
 * Bonded particles -- i.e. next neighbors in an arm, the central core particle
 * and the particles attached to it, or the functionalized particles with their
 * direct neighbor in the corresponding arm -- interact via the
 * `OpenMPCD::PairPotentials::FENE` potential.
 * The interaction parameters `K`, `l_0` and `R` are set to \f$ K_{\mu\nu} \f$,
 * \f$ l_{\mu\nu} \f$, and \f$ R_{\mu\nu} \f$, respectively, which will be
 * described below. The indices \f$ \mu \f$ and \f$ \nu \f$ denote between which
 * kinds of particles (i.e. core, arm, or magnetic) the interaction effects a
 * force.
 *
 * Finally, each pair of particles interacts via the
 * `OpenMPCD::PairPotentials::WeeksChandlerAndersen_DistanceOffset` (WCA)
 * potential.
 * The interaction parameters `epsilon`, `sigma`, and `d` are set to
 * \f$ \varepsilon_{\mu\nu} \f$, \f$ \sigma_{\mu\nu} \f$, and
 * \f$ D_{\mu\nu} \f$, respectively, which will be described below.
 * The indices \f$ \mu \f$ and \f$ \nu \f$ have the same meaning as with the
 * FENE potential.
 *
 * In the configuration group `interactionParameters`, one has to set
 * `epsilon_core`, `epsilon_arm`, and `epsilon_magnetic`, which will be referred
 * to as \f$ \varepsilon_C \f$, \f$ \varepsilon_A \f$, and
 * \f$ \varepsilon_M \f$, respectively.
 * Similarly, the properties `sigma_core`, `sigma_arm`, and `sigma_magnetic`
 * define \f$ \sigma_C \f$, \f$ \sigma_A \f$, and \f$ \sigma_M \f$, and
 * `D_core`, `D_arm`, and `D_magnetic` define \f$ D_C \f$, \f$ D_A \f$, and
 * \f$ D_M \f$.
 * Finally, one also has to set `magneticPrefactor` in the
 * `interactionParameters` group. The `dipoleOrientation` array, if not set,
 * will be assumed to be `[0, 0, 1]`.
 *
 * From these quantities, the values for \f$ \varepsilon_{\mu\nu} \f$,
 * \f$ \sigma{\mu\nu} \f$, and \f$ D_{\mu\nu} \f$, where each \f$ \mu \f$ and
 * \f$ \nu \f$ can take on the values \f$ C \f$ for core particles, \f$ A \f$
 * for non-functionalized arm particles, and \f$ M \f$ for functionalized
 * magnetic particles, can be calculated according to the Lorentz-Berthelot
 * mixing rules:
 * \f[ \varepsilon_{\mu\nu} = \sqrt{ \varepsilon_\mu \varepsilon_\nu } \f]
 * \f[ \sigma_{\mu\nu} = \frac{ \sigma_\mu + \sigma_\nu }{ 2 } \f]
 * \f[ D_{\mu\nu} = \frac{ D_\mu + D_\nu }{ 2 } \f]
 *
 * Finally, the parameters \f$ l_{\mu\nu} \f$, \f$ R_{\mu\nu} \f$, and
 * \f$ K_{\mu\nu} \f$, are defined via
 * \f[ l_{\mu\nu} = D_{\mu\nu} \f]
 * \f[ R_{\mu\nu} = 1.5 \sigma_{\mu\nu} \f]
 * \f[ K_{\mu\nu} = 30 \varepsilon_{\mu\nu} \sigma_{\mu\nu}^{-2} \f]
 *
 *
 * In addition to the `interactionParmeters` settings group, the configuration
 * passed to the constructor is expected to contain the `structure` group,
 * which contains in `starCount` the number of stars, in `armCountPerStar` how
 * many arms each star has, in `armParticlesPerArm` how many ordinary arm
 * particles there are in each arm, and in `hasMagneticParticles` whether
 * each arm should be terminated with a magnetic particle (in addition
 * to the ordinary arm particles). Furthermore, `particleMass` contains the mass
 * each individual particle has.
 *
 * Furthermore, one must set `mdTimeStepSize` to the number of MPC time units
 * one should advance per MD time-step, which must be a positive number.
 *
 * If the configuration contains the value `initialState`, its value is
 * taken as a path of a VTF snapshot file (see `OpenMPCD::VTFSnapshotFile`) that
 * specifies the initial configuration of the stars. If the snapshot does not
 * contain velocity information, initial velocities are assumed to be zero.
 *
 * @tparam PositionCoordinate The type to store position coordinates.
 * @tparam VelocityCoordinate The type to store velocity coordinates.
 */
template<
	typename PositionCoordinate = MPCParticlePositionType,
	typename VelocityCoordinate = MPCParticleVelocityType>
class StarPolymers
	: public MPCSolute::Base<PositionCoordinate, VelocityCoordinate>
{
public:
	typedef PositionCoordinate ForceCoordinate;
		///< The type to store force coordinates.

	typedef ImplementationDetails::StarPolymers::ParticleType ParticleType;
		///< Holds the enumeration of particle types.

public:
	/**
	 * The constructor.
	 *
	 * @throw OpenMPCD::InvalidConfigurationException
	 *        Throws if the configuration is not valid.
	 *
	 * @param[in] settings
	 *            The configuration for this instance.
	 * @param[in] boundaryCondition
	 *            The boundary conditions that have been configured. Currently,
	 *            only LeesEdwards boundary conditions are supported.
	 */
	StarPolymers(
		const Configuration::Setting& settings,
		const BoundaryCondition::Base* const boundaryCondition);

private:
	StarPolymers(const StarPolymers&); ///< The copy constructor.

public:
	/**
	 * The destructor.
	 */
	virtual ~StarPolymers();

public:
	/**
	 * Performs, on the Device, an MD timestep of size `getMDTimeStepSize()`.
	 */
	void performMDTimestep();

	/**
	 * Returns the total number of polymer stars in this instance.
	 */
	std::size_t getStarCount() const
	{
		return 1;
	}

	/**
	 * Returns the number of arms per star.
	 */
	std::size_t getArmCountPerStar() const;

	/**
	 * Returns the number of particles per arm.
	 */
	std::size_t getParticleCountPerArm() const;

	/**
	 * Returns whether arms have an additional magnetic particle attached.
	 */
	bool hasMagneticParticles() const;

	/**
	 * Returns the number of particles per arm, including magnetic particles.
	 */
	std::size_t getParticleCountPerArmIncludingMagneticParticles() const;

	/**
	 * Returns the total number of individual particles per star.
	 */
	std::size_t getParticleCountPerStar() const;

	/**
	 * Returns the total number of individual particles in all stars.
	 */
	std::size_t getParticleCount() const;

	/**
	 * Returns the number of logical entities in the solute.
	 */
	std::size_t getNumberOfLogicalEntities() const
	{
		return getStarCount();
	}

	/**
	 * Returns the mass of a particle, which is assumed to be equal for all
	 * particles in this instance.
	 */
	virtual FP getParticleMass() const
	{
		return particleMass;
	}

	/**
	 * Writes structure information to the given snapshot file.
	 *
	 * @throw OpenMPCD::NULLPointerException
	 *        Throws if `snapshot == nullptr`.
	 * @throw OpenMPCD::InvalidArgumentException
	 *        Throws if the given snapshot is not in write mode, or if the
	 *        structure block already has been processed.
	 *
	 * @param[in,out] snapshot The snapshot file.
	 */
	void writeStructureToSnapshot(VTFSnapshotFile* const snapshot) const;

	/**
	 * Writes the particle positions and velocities to the given snapshot file.
	 *
	 * This function will call `fetchFromDevice`, and write the current Device
	 * data to the given snapshot file.
	 *
	 * @throw OpenMPCD::NULLPointerException
	 *        Throws if `snapshot == nullptr`.
	 * @throw OpenMPCD::InvalidArgumentException
	 *        Throws if the number of atoms declared in the snapshot does not
	 *        match the number of particles in this instance.
	 * @throw OpenMPCD::InvalidArgumentException
	 *        Throws if the given snapshot is not in write mode.
	 *
	 * @param[in,out] snapshot The snapshot file.
	 */
	void writeStateToSnapshot(VTFSnapshotFile* const snapshot) const;

	/**
	 * Reads the next timestep block from the given snapshot file, and uses it
	 * as the current state.
	 *
	 * If no velocities are specified, the current ones are kept.
	 *
	 * @throw OpenMPCD::NULLPointerException
	 *        If `OPENMPCD_DEBUG` is defined, throws if `snapshot == nullptr`.
	 * @throw OpenMPCD::InvalidArgumentException
	 *        Throws if `!snapshotStructureIsCompatible()`.
	 *
	 * @param[in,out] snapshot The snapshot file to read from.
	 */
	void readStateFromSnapshot(VTFSnapshotFile* const snapshot);

	/**
	 * Returns whether the given snapshot file is compatible with the star
	 * polymer architecture, e.g. whether the number of stars, arms per star,
	 * etc. match.
	 *
	 * If `!snapshot.isInReadMode()`, `false` is returned.
	 *
	 * @param[in] snapshot The snapshot to check for compatibility.
	 */
	bool snapshotStructureIsCompatible(const VTFSnapshotFile& snapshot) const;

private:
	StarPolymers& operator=(const StarPolymers); ///< The assignment operator.

private:
	/**
	 * Returns whether the given particle ID is valid, i.e. whether
	 * `particleID < getParticleCount()`.
	 *
	 * @param[in] particleID The particle ID to query.
	 */
	bool isValidParticleID(const std::size_t particleID) const;

	/**
	 * Returns the particle type that corresponds to the given ID.
	 *
	 * @throw OpenMPCD::OutOfBoundsException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `!isValidParticleID(particleID)`.
	 *
	 * @param[in] particleID The particle ID to get the type for.
	 */
	typename ParticleType::Enum
	getParticleType(const std::size_t particleID) const;

	/**
	 * Returns the particle ID for the given input.
	 *
	 * Use of this function implies that the particle in question is part of an
	 * arm.
	 *
	 * @throw OpenMPCD::OutOfBoundsException
	 *        If `OPENMPCD_DEBUG` is defined, throws if `star >= getStarCount()`.
	 * @throw OpenMPCD::OutOfBoundsException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `arm >= getArmCountPerStar()`.
	 * @throw OpenMPCD::OutOfBoundsException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `particleInArm >= getParticleCountPerArm()`.
	 *
	 * @param[in] star          The ID of the star, starting at `0`.
	 * @param[in] arm           The ID of the arm, starting at `0`.
	 * @param[in] particleInArm The ID of the particle in the arm, starting at
	 *                          `0`.
	 */
	std::size_t getParticleID(
		const std::size_t star, const std::size_t arm,
		const std::size_t particleInArm) const;

	/**
	 * Returns the particle ID for the given input.
	 *
	 * Use of this function implies that the particle in question is a magnetic
	 * particle.
	 *
	 * @throw OpenMPCD::OutOfBoundsException
	 *        If `OPENMPCD_DEBUG` is defined, throws if `star >= getStarCount()`.
	 * @throw OpenMPCD::OutOfBoundsException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `arm >= getArmCountPerStar()`.
	 *
	 * @param[in] star The ID of the star, starting at `0`.
	 * @param[in] arm  The ID of the arm, starting at `0`.
	 */
	std::size_t getParticleID(
		const std::size_t star, const std::size_t arm) const;

	/**
	 * Returns the particle ID for the given input.
	 *
	 * Use of this function implies that the particle in question is a core
	 * particle.
	 *
	 * @throw OpenMPCD::OutOfBoundsException
	 *        If `OPENMPCD_DEBUG` is defined, throws if `star >= getStarCount()`.
	 *
	 * @param[in] star The ID of the star, starting at `0`.
	 */
	std::size_t getParticleID(const std::size_t star) const;


	/**
	 * Initializes the positions on the Host.
	 */
	void initializePositionsOnHost();

	/**
	 * Initializes the velocities on the Host.
	 */
	void initializeVelocitiesOnHost();

private:
	std::size_t starCount;          ///< The number of stars.
	std::size_t armCountPerStar;    ///< The number of arms per star.
	std::size_t armParticlesPerArm; ///< The number of particles per arm.
	bool hasMagneticParticles_;     ///< Whether the arms have magnetic ends.

	FP particleMass; ///< The mass of any one particle.

	ForceCoordinate* d_forces1; ///< Work buffer storing forces.
	ForceCoordinate* d_forces2; ///< Work buffer storing forces.

	PairPotentials::WeeksChandlerAndersen_DistanceOffset<ForceCoordinate>**
		wcaPotentials; ///< The general particle-particle interactions.
	PairPotentials::FENE<ForceCoordinate>**
		fenePotentials; ///< The interactions for bonded pairs.
	PairPotentials::
		MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<
			ForceCoordinate>**
				magneticPotential;
					///< The interactions for pairs of magnetic particles.

private:
	using Base<PositionCoordinate, VelocityCoordinate>::fetchFromDevice;
	using Base<PositionCoordinate, VelocityCoordinate>::getMDTimeStepSize;
	using Base<PositionCoordinate, VelocityCoordinate>::pushToDevice;

	using Base<PositionCoordinate, VelocityCoordinate>::deviceMemoryManager;
	using Base<PositionCoordinate, VelocityCoordinate>::mdTimeStepSize;
	using Base<PositionCoordinate, VelocityCoordinate>::h_positions;
	using Base<PositionCoordinate, VelocityCoordinate>::h_velocities;
	using Base<PositionCoordinate, VelocityCoordinate>::d_positions;
	using Base<PositionCoordinate, VelocityCoordinate>::d_velocities;
}; //class StarPolymers

} //namespace MPCSolute
} //namespace CUDA
} //namespace OpenMPCD

#endif //OPENMPCD_CUDA_MPCSOLUTE_STARPOLYMERS_HPP
