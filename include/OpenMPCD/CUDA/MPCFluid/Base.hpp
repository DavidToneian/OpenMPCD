/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Base class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_BASE_HPP
#define OPENMPCD_CUDA_MPCFLUID_BASE_HPP

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/Base.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/RemotelyStoredVector.hpp>
#include <OpenMPCD/Vector3D.hpp>
#include <OpenMPCD/VTFSnapshotFile.hpp>

namespace OpenMPCD
{
namespace CUDA
{
	class Simulation;

/**
 * Namespace for MPC Fluid classes.
 */
namespace MPCFluid
{

	/**
	 * Base class for MPC fluids.
	 *
	 * The MPC fluid consists of a certain number of <em>logical entities</em>,
	 * each of which consists of one or more <em>MPC particles</em>.
	 *
	 * Examples of logical entities include linear polymers, ring polymers, and
	 * dendrimers. The MPC particles (i.e. the fundamental objects that MPC is
	 * concerned with) that would make up those logical entities would typically
	 * couple to each other via interaction potentials that depend on the kind
	 * of MPC fluid at hand.
	 */
	class Base
	{
		friend class CUDA::Simulation;

		protected:
			/**
			 * The constructor.
			 * This constructor neither initializes the fluid state on Host or Device,
			 * nor does it synchronize Host and Device.
			 * @param[in] sim                The simulation instance.
			 * @param[in] count              The number of fluid particles.
			 * @param[in] streamingTimestep_ The timestep for a streaming step.
			 * @param[in] rng_               A random number generator to seed this instance's RNG with.
			 * @param[in] devMemMgr          The Device memory manager.
			 */
			Base(
				const CUDA::Simulation* const sim, const unsigned int count, const FP streamingTimestep_,
				RNG& rng_, DeviceMemoryManager* const devMemMgr);

		private:
			Base(const Base&); ///< The copy constructor.

		public:
			/**
			 * The destructor.
			 */
			virtual ~Base();

		public:
			/**
			 * Returns the number of MPC fluid particles.
			 */
			unsigned int getParticleCount() const
			{
				return numberOfParticles;
			}

			/**
			 * Returns the MPC fluid particle mass.
			 */
			static unsigned int getParticleMass()
			{
				return mpcParticleMass;
			}

			/**
			 * Returns the number of logical entities in the fluid.
			 */
			virtual unsigned int getNumberOfLogicalEntities() const = 0;

			/**
			 * Returns whether all logical entities consist of the same number
			 * of MPC particles.
			 *
			 * The value of this function will remain unchanged for the lifetime
			 * of the instance.
			 */
			virtual
			bool numberOfParticlesPerLogicalEntityIsConstant() const = 0;

			/**
			 * Returns the number of MPC particles per logical entity.
			 *
			 * @throw OpenMPCD::InvalidCallException
			 *        Throws if
			 *        `!numberOfParticlesPerLogicalEntityIsConstant()`.
			 */
			virtual unsigned int getNumberOfParticlesPerLogicalEntity() const;

			/**
			 * Copies the MPC fluid particles from the CUDA Device to the Host.
			 */
			void fetchFromDevice() const;

			/**
			 * Returns a MPC fluid particle's position vector.
			 * @warning
			 * This function only returns the position that was current the last time
			 * fetchFromDevice was called.
			 * @throw OutOfBoundsException If OPENMPCD_DEBUG is defined,
			 *                             throws if particleID >= getMPCParticleCount()
			 * @param[in] particleID The particle ID.
			 */
			const RemotelyStoredVector<const MPCParticlePositionType>
				getPosition(const unsigned int particleID) const;

			/**
			 * Returns a MPC fluid particle's velocity vector.
			 * @warning
			 * This function only returns the velocity that was current the last time
			 * fetchFromDevice was called.
			 * @throw OutOfBoundsException If OPENMPCD_DEBUG is defined,
			 *                             throws if particleID >= getMPCParticleCount()
			 * @param[in] particleID The particle ID.
			 */
			const RemotelyStoredVector<const MPCParticleVelocityType>
				getVelocity(const unsigned int particleID) const;

			/**
			 * Sets the positions and velocities of the particles on the Device.
			 *
			 * @throw OpenMPCD::NULLPointerException
			 *        If `OPENMPCD_DEBUG` is defined, throws if
			 *        `positions == nullptr` or `velocities == nullptr`.
			 *
			 * @param[in] positions  An array holding `3 * getParticleCount()`
			 *                       values on the Host; first, the `x`, `y`,
			 *                       and `z` coordinates of the first atom, then
			 *                       those of the second, etc.
			 * @param[in] velocities An array holding `3 * getParticleCount()`
			 *                       values on the Host; first, the `x`, `y`,
			 *                       and `z` velocities of the first atom, then
			 *                       those of the second, etc.
			 */
			void setPositionsAndVelocities(
				const MPCParticlePositionType* const positions,
				const MPCParticleVelocityType* const velocities);

			/**
			 * Returns a const pointer to the MPC fluid positions on the Device.
			 */
			const MPCParticlePositionType* getDevicePositions() const
			{
				return d_mpcParticlePositions;
			}

			/**
			 * Returns a pointer to the MPC fluid positions on the Device.
			 */
			MPCParticlePositionType* getDevicePositions()
			{
				return d_mpcParticlePositions;
			}

			/**
			 * Returns a pointer to the MPC fluid velocities on the Device.
			 */
			MPCParticleVelocityType* getDeviceVelocities()
			{
				return d_mpcParticleVelocities;
			}

			/**
			 * Returns a const pointer to the MPC fluid velocities on the Device.
			 */
			const MPCParticleVelocityType* getDeviceVelocities() const
			{
				return d_mpcParticleVelocities;
			}

			/**
			 * Returns a pointer to the MPC fluid positions on the Host.
			 */
			MPCParticlePositionType* getHostPositions()
			{
				return mpcParticlePositions;
			}

			/**
			 * Returns a pointer to the MPC fluid velocities on the Host.
			 */
			MPCParticleVelocityType* getHostVelocities()
			{
				return mpcParticleVelocities;
			}

			/**
			 * Returns the fluid instrumentation.
			 * @throw NULLPointerException If OPENMPCD_DEBUG is defined, throws if instrumentation is a NULL pointer.
			 */
			Instrumentation::Base& getInstrumentation() const
			{
				#ifdef OPENMPCD_DEBUG
					if(instrumentation==NULL)
						OPENMPCD_THROW(NULLPointerException, "instrumentation");
				#endif

				return *const_cast<Instrumentation::Base*>(instrumentation);
			}

			/**
			 * Writes the particle positions and velocities to the given
			 * snapshot file.
			 *
			 * This function will call `fetchFromDevice`, and write the current
			 * Device data to the given snapshot file.
			 *
			 * @throw OpenMPCD::NULLPointerException
			 *        Throws if `snapshot == nullptr`.
			 * @throw OpenMPCD::InvalidArgumentException
			 *        Throws if the number of atoms declared in the snapshot
			 *        does not match the number of particles in this instance.
			 * @throw OpenMPCD::InvalidArgumentException
			 *        Throws if the given snapshot is not in write mode.
			 *
			 * @param[in,out] snapshot The snapshot file.
			 */
			void writeToSnapshot(VTFSnapshotFile* const snapshot) const;

			/**
			 * Computes, on the Host, which MPC particles match the criterion represented by `func`.
			 * @warning
			 * This function operates only on the data that were current the last time `fetchFromDevice` was called.
			 * @param[in]  func       Iff this function returns true, the MPC particles are considered to satisfy
			 *                        the criterion. The first argument is the particle's position, the second
			 *                        its velocity.
			 * @param[out] matches    Will hold the IDs of the particles that satisfied the criterion; may be `nullptr` if not needed.
			 * @param[out] matchCount Will hold the number of particles that satisfied the criterion; may be `nullptr` if not needed.
			 */
			void findMatchingParticlesOnHost(
				bool (*func)(const RemotelyStoredVector<const MPCParticlePositionType>&, const RemotelyStoredVector<const MPCParticleVelocityType>&),
				std::vector<unsigned int>* const matches,
				unsigned int* const matchCount
				) const;


			/**
			 * Computes the center of mass for each logical entity, and saves
			 * their coordinates in the given Device buffer.
			 *
			 * The computation is performed with the data as it currently exists
			 * on the Device.
			 *
			 * @throw OpenMPCD::NULLPointerException
			 *        Throws if `buffer == nullptr`.			 *
			 *
			 * @param[out] buffer The device buffer to save the coordinates to.
			 *                    It must be able to hold at least
			 *                    `3 * getNumberOfLogicalEntities()` elements.
			 *                    The first element in the buffer will be
			 *                    the `x` coordinate of the center of mass of
			 *                    the first logical entity, followed by the `y`
			 *                    and `z` coordinates. After that, the second
			 *                    entity's coordinates follow, and so on.
			 */
			void saveLogicalEntityCentersOfMassToDeviceMemory(
				MPCParticlePositionType* const buffer) const;

		protected:
			/**
			 * Performs a streaming step.
			 */
			virtual void stream() = 0;

			/**
			 * Copies the MPC fluid particles from the Host to the CUDA Device.
			 */
			void pushToDevice();

			/**
			 * Initializes the fluid particles velocities in Host memory.
			 */
			void initializeVelocitiesOnHost() const;

			/**
			 * Scales all MPC fluid particle velocities so that the global temperature matches the given target.
			 * @param[in]     kT            The target temperature, multiplied by Boltzmann's constant.
			 * @param[in,out] velocities    The MPC fluid particle velocities on the Host.
			 * @param[in]     particleCount The number of MPC fluid particles.
			 */
			static void globalUnbiasedThermostat(const FP kT, MPCParticleVelocityType* const velocities,
			                                     const unsigned int particleCount);

			/**
			 * Returns the total momentum of the given MPC fluid particles.
			 * @param[in] velocities    The velocities of the MPC fluid particles on the Host.
			 * @param[in] particleCount The number of MPC fluid particles.
			 */
			static const Vector3D<MPCParticleVelocityType>
				getTotalMomentum(const MPCParticleVelocityType* const velocities,
			                     const unsigned int particleCount);

			/**
			 * Returns the mean momentum of the given MPC fluid particles.
			 * @param[in] velocities    The velocities of the MPC fluid particles on the Host.
			 * @param[in] particleCount The number of MPC fluid particles.
			 */
			static const Vector3D<MPCParticleVelocityType>
				getMeanMomentum(const MPCParticleVelocityType* const velocities,
			                    const unsigned int particleCount);

			/**
			 * Returns the total kinetic energy of the MPC fluid.
			 * @param[in] velocities    The velocities of the MPC fluid particles on the Host.
			 * @param[in] particleCount The number of MPC fluid particles.
			 */
			static FP getKineticEnergy(const MPCParticleVelocityType* const velocities,
			                           const unsigned int particleCount);

			/**
			 * Returns the product of Boltzmann's constant with the MPC fluid temperature,
			 * as measured by the total kinetic energy of the MPC fluid.
			 * @param[in] velocities    The velocities of the MPC fluid particles on the Host.
			 * @param[in] particleCount The number of MPC fluid particles.
			 */
			static FP getkTViaKineticEnergy(const MPCParticleVelocityType* const velocities,
			                                const unsigned int particleCount)
			{
				return 2.0/3.0*getKineticEnergy(velocities, particleCount)/particleCount;
			}

		private:
			/**
			 * Reads the configuration.
			 */
			void readConfiguration();

		private:
			const Base& operator=(const Base&); ///< The assignment operator.

		protected:
			static const unsigned int mpcParticleMass = 1; ///< The mass of each MPC particle.

			const CUDA::Simulation* const simulation; ///< The simulation instance.
			DeviceMemoryManager* const deviceMemoryManager; ///< The Device memory manager.

			Instrumentation::Base* instrumentation; ///< The fluid instrumentation.

			mutable RNG rng; ///< The random number generator.

			mutable MPCParticlePositionType* mpcParticlePositions; ///< Host pointer for the MPC particle positions.
			mutable MPCParticleVelocityType* mpcParticleVelocities; ///< Host pointer for the MPC particle positions.

			MPCParticlePositionType* d_mpcParticlePositions; ///< Device pointer for the MPC particle positions.
			MPCParticleVelocityType* d_mpcParticleVelocities; ///< Device pointer for the MPC particle positions.

			const unsigned int numberOfParticles; ///< The number of fluid particles.

			const FP streamingTimestep; ///< The timestep for a streaming step.
	};

} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
