/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Instrumentation::VelocityAutocorrelation::Base class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_VELOCITYAUTOCORRELATION_BASE_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_VELOCITYAUTOCORRELATION_BASE_HPP

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/Vector3D.hpp>

#include <boost/tuple/tuple.hpp>
#include <deque>
#include <vector>

namespace OpenMPCD
{
namespace CUDA
{

class Simulation;

namespace MPCFluid
{

class Base;

namespace Instrumentation
{
/**
 * Namespace for classes that measure the velocity autocorrelation in MPC fluids.
 */
namespace VelocityAutocorrelation
{
/**
 * Base class for measurements of velocity autocorrelation in MPC fluids.
 *
 * This class and its subclasses aid in measuring the velocity autocorrelation
 * function of the fluid constituents' centers of mass, i.e.
 * \f[
 * 		\left< \vec{v} \left( t \right) \cdot \vec{v} \left( 0 \right) \right>
 * 		=
 * 		\frac{ 1 }{ N_{\textrm{logical entites}} }
 * 		\sum_{k = 0}^{N_{\textrm{logical entites}} - 1}
 * 		\left<
 * 			\vec{v}_k \left( t \right) \cdot \vec{v}_k \left( 0 \right)
 * 		\right>
 * \f]
 * where \f$ N_{\textrm{logical entites}} \f$ is the number of logical entities
 * in the fluid (see `OpenMPCD::CUDA::MPCFluid::Base`),
 * \f$ \vec{v}_k \left( t \right) \f$ is the velocity of the center of
 * mass of the logical entity with index \f$ k \f$ at simulation time
 * \f$ t \f$, and the angle brackets denote expectation values.
 *
 * To approximate this quantity, the center of mass velocities of the logical
 * entities are stored in snapshots periodically.
 * Then, each time `measure` is called, the quantity
 * \f[
 * 		V \left( t_i, t \right)
 * 		=
 * 		\frac{ 1 }{ N_{\textrm{logical entites}} }
 * 		\sum_{k = 0}^{N_{\textrm{logical entites}} - 1}
 * 		\vec{v}_k \left( t_i \right)
 * 		\cdot
 *		\vec{v}_k \left( t \right)
 * \f]
 * is calculated, i.e. the sample average of the inner product of a logical
 * entity's center of mass velocity at the current time \f$ t \f$ with that same
 * logical entity's center of mass velocity at snapshot time
 * \f$ t_i \f$, for all currently available snapshots \f$ i \f$.
 *
 * At most, `snapshotCount` (see configuration options below) many snapshots are
 * stored at any point in time.
 * Every time `measure` is called, before the \f$ V \left( t_i, t \right) \f$
 * are calculated as described above, the following procedure takes place:
 *  - If no snapshots have been taken yet, a snapshot it taken.
 *  - Otherwise, if fewer than `snapshotCount` snapshots have already been
 *    created, the latest snapshot time is subtracted from the current
 *    simulation time. If that difference is larger than
 *    `measurementTime / snapshotCount - 1e-6`, a new snapshot is taken and
 *    stored after the previously taken snapshot.
 *  - If there have already been `snapshotCount` many snapshots before the call
 *    to `measure`, all of the snapshots are iterated over. For each such
 *    snapshot, if the difference between the current simulation time and that
 *    snapshot's time is greater than `measurementTime + 1e-6`, that snapshot is
 *    overridden in place with the current state and simulation time.
 * The order of snapshots in memory, as described above, is of consequence for
 * the order in which measurements are written by the `save` function.
 *
 * Given this, the velocity autocorrelation function
 * \f$
 * 		\left< \vec{v} \left( t \right) \cdot \vec{v} \left( 0 \right) \right>
 * \f$
 * can be approximated as the average of all \f$ V \left( t_i, t_j \right) \f$
 * where \f$ t_j - t_i = t \f$.
 *
 * However, this last step is currently not performed by this class;
 * instead, it computes \f$ V \left( t_i, t_j \right) \f$ and stores them, along
 * with \f$ t_i \f$ and \f$ t_j \f$.
 *
 * This class and its subclasses are configured via the
 * `instrumentation.velocityAutocorrelation` configuration group, which consists
 * of the following settings:
 *  - `snapshotCount`, a positive integer, determines the maximum number
 *     of snapshots stored in memory at any given time.
 *  - `measurementTime`, which is a floating-point value that is approximately
 *     the maximal time for which the correlation function is measured;
 *     see above for details.
 */
class Base
{
	protected:
		/**
		 * Represents a snapshot of the MPC fluid constituent velocities.
		 */
		class Snapshot
		{
			public:
				/**
				 * The constructor.
				 * @param[in] numberOfConstituents The number of logical constituents (e.g. pairs, triplets, ...)
				 *                                 in the fluid.
				 * @param[in] devMemMgr            The Device memory manager.
				 */
				Snapshot(const unsigned int numberOfConstituents, DeviceMemoryManager* const devMemMgr);

			private:
				Snapshot(const Snapshot&); ///< The copy constructor.

			public:
				/**
				 * The destructor.
				 */
				~Snapshot();

			public:
				/**
				 * Returns the MPC timestamp of the last snapshot.
				 */
				FP getSnapshotTime() const
				{
					return snapshotTime;
				}

				/**
				 * Sets the MPC timestamp of the last snapshot.
				 * @param[in] time The time this snapshot has last been updated.
				 */
				void setSnapshotTime(const FP time)
				{
					snapshotTime = time;
				}

				/**
				 * Returns the Device buffer used to the save MPC fluid constituent velocities.
				 */
				MPCParticleVelocityType* getBuffer()
				{
					return buffer;
				}

			private:
				const Snapshot& operator=(const Snapshot&); ///< The assignment operator.

			private:
				FP snapshotTime; ///< The time this snapshot was taken.
				MPCParticleVelocityType* buffer; ///< The Device buffer for the MPC fluid constituent velocities.
				DeviceMemoryManager* const deviceMemoryManager; ///< The simulation instance.
		}; //class Snapshot

	protected:
		/**
		 * The constructor.
		 *
		 * @throw OpenMPCD::InvalidConfigurationException
		 *        Throws if the configuration is invalid.
		 * @throw OpenMPCD::NULLPointerException
		 *        If `OPENMPCD_DEBUG` is defined, throws if any of the pointer
		 *        arguments is a `nullptr`.
		 * @throw OpenMPCD::InvalidArgumentException
		 *        If `OPENMPCD_DEBUG` is defined, throws if
		 *        `numberOfConstituents_ == 0`.
		 *
		 * @param[in] sim                   The simulation instance.
		 * @param[in] devMemMgr             The Device memory manager instance.
		 * @param[in] mpcFluid_             The MPC fluid to measure.
		 * @param[in] numberOfConstituents_ The number of logical constituents (e.g. pairs, triplets, ...) in the fluid.
		 */
		Base(
			const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
			const MPCFluid::Base* const mpcFluid_,
			const unsigned int numberOfConstituents_);

	private:
		Base(const Base&); ///< The copy constructor.

	public:
		/**
		 * The destructor.
		 */
		virtual ~Base();

	public:
		/**
		 * Performs measurements.
		 */
		void measure();

		/**
		 * Saves the data to the given run directory.
		 *
		 * The given run directory is created if necessary, and so is the data
		 * file therein; if the data file already exists, it will be truncated.
		 *
		 * Multiple lines will be written to the file, each consisting of three
		 * tab-separated columns:
		 * The first column corresponds to the simulation time
		 * \f$ t_i \f$ of a snapshot \f$ i \f$; the second
		 * column to the simulation time \f$ t_j \ge t_i \f$ ; and finally,
		 * the third column contains \f$ V \left( t_i, t_j \right) \f$.
		 * One such line will be written for every measured \f$ V \f$ (see class
		 * documentation for details).
		 *
		 * The lines are ordered by ascending \f$ t_j \f$
		 * first, and then by ascending snapshot position in memory (see class
		 * documentation for details).
		 *
		 * @param[in] rundir
		 *            The path to the run directory. Within that directory,
		 *            the data will be saved to the file
		 *            `velocityAutocorrelations.data`.
		 */
		void save(const std::string& rundir) const;

	public:
		/**
		 * Returns whether the given simulation configured this instrumentation.
		 * @param[in] sim The simulation instance.
		 * @throw NULLPointerException Throws if `sim` is `nullptr`.
		 */
		static bool isConfigured(const Simulation* const sim);

	protected:
		/**
		 * Populates the Device buffer with the current velocities of the fluid constituents.
		 */
		virtual void populateCurrentVelocities() = 0;

	private:
		/**
		 * Reads the configuration.
		 */
		void readConfig();

		/**
		 * Updates the snapshots as necessary.
		 */
		void updateSnapshots();

		/**
		 * Updates the given snapshot.
		 * @param[in] snapshot The snapshot to update.
		 */
		void updateSnapshot(Snapshot* const snapshot);

		/**
		 * Checks whether the given snapshot should be initialized, and does so if appropriate.
		 * @param[in] snapshot The snapshot index.
		 * @return Returns true if the snapshot has been initialized (possibly during previous calls)
		 *         and is ready for measurement, false otherwise.
		 */
		bool initializeSnapshotIfAppropriate(const unsigned int snapshotID);

		/**
		 * Performs a measurement for the given snapshot.
		 * @param[in] snapshot The snapshot to measure with.
		 */
		void measureWithSnapshot(Snapshot* const snapshot);

		/**
		 * Returns the value of the correlation function for the given snapshot,
		 * performing all calculations on the CPU in a naive way.
		 * Note that this function is intended to be used as a verification of the GPU
		 * result, and it is in no way optimized for performance.
		 * @param[in] snapshot The snapshot to measure with.
		 */
		MPCParticleVelocityType getCorrelationForSnapshot_CPU_naive(Snapshot* const snapshot);

		/**
		 * Returns the value of the correlation function for the given snapshot,
		 * performing all calculations on the CPU using the thrust library.
		 * Note that this function is intended to be used as a verification of the GPU
		 * result, and it is in no way optimized for performance.
		 * @param[in] snapshot The snapshot to measure with.
		 */
		MPCParticleVelocityType getCorrelationForSnapshot_CPU_thrust(Snapshot* const snapshot);

		/**
		 * Saves the data to the given file.
		 *
		 * See `save` for details on the output format.
		 *
		 * @param[in] path The file path to save the data to.
		 */
		void saveAutocorrelations(const std::string& path) const;

	protected:
		const Simulation* const simulation; ///< The simulation instance.
		DeviceMemoryManager* const deviceMemoryManager; ///< The Device memory manager.
		const MPCFluid::Base* const mpcFluid; ///< The fluid to measure.

		const unsigned int numberOfConstituents;
			///< The number of logical constituents (e.g. pairs, triplets, ...) in the fluid.

		FP measurementTime; ///< The maximum time between snapshots.

		std::vector<Snapshot*> snapshots; ///< The snapshots of the fluid constituent velocities.
		MPCParticleVelocityType* currentVelocities; ///< Current velocities of the fluid constituents.

		std::deque<boost::tuple<FP, FP, MPCParticleVelocityType> > autocorrelations;
			///< Collection of tuples consisting of, in that order,
			///  \f$ t_i \f$, \f$ t_j \f$, and
			///  \f$ V \left( t_i, t_j \right) \f$.
}; //class Base
} //namespace VelocityAutocorrelation
} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
