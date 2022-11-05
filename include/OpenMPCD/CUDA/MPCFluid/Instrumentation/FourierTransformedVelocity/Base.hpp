/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Instrumentation::FourierTransformedVelocity::Base class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_FOURIERTRANSFORMEDVELOCITY_BASE_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_FOURIERTRANSFORMEDVELOCITY_BASE_HPP

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/Vector3D.hpp>

#include <complex>
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
 * Namespace for classes that measure the Fourier-transformed velocities in MPC fluids.
 *
 * See @ref velocityInFourierSpace
 */
namespace FourierTransformedVelocity
{
/**
 * Base class for measurements of Fourier-transformed velocities in MPC fluids.
 */
class Base
{
	protected:
		/**
		 * The constructor.
		 * @param[in] sim                  The simulation instance.
		 * @param[in] devMemMgr            The Device memory manager.
		 * @param[in] mpcFluid_            The MPC fluid to measure.
		 * @param[in] numberOfConstituents The number of logical constituents (e.g. pairs, triplets, ...) in the fluid.
		 */
		Base(
			const Simulation* const sim, DeviceMemoryManager* devMemMgr,
			const MPCFluid::Base* const mpcFluid_,
			const unsigned int numberOfConstituents);

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
		virtual void measure() = 0;

		/**
		 * Saves the data to the given run directory.
		 * @param[in] rundir The path to the run directory.
		 */
		virtual void save(const std::string& rundir) const;

	public:
		/**
		 * Returns whether the given simulation configured this instrumentation.
		 * @param[in] sim The simulation instance.
		 * @throw NULLPointerException Throws if `sim` is `nullptr`.
		 */
		static bool isConfigured(const Simulation* const sim);

	private:
		/**
		 * Reads the configuration.
		 */
		void readConfig();

		/**
		 * Saves the Fourier-transformed velocities to the given directory.
		 * @param[in] datadir The directory to save the data in.
		 */
		void saveFourierTransformedVelocities(const std::string& datadir) const;

		/**
		 * Prints the Fourier-transformed velocities to the given stream.
		 * @param[in] k      The index for the \f$\vec{k}\f$ vector.
		 * @param[in] stream The stream to print to.
		 * @throw OutOfBoundsException Throws if index is out of bounds.
		 */
		void saveFourierTransformedVelocities(const unsigned int index, std::ostream& stream) const;

	protected:
		const Simulation* const simulation; ///< The simulation instance.
		DeviceMemoryManager* const deviceMemoryManager; ///< The Device memory manager.
		const MPCFluid::Base* const mpcFluid; ///< The fluid to measure.

		MPCParticleVelocityType* realBuffer; ///< Device buffer for temporary results.
		MPCParticleVelocityType* imagBuffer; ///< Device buffer for temporary results.

		std::vector<std::vector<MPCParticlePositionType> > k_n;
			///< The multiples of \f$ 2*\pi/L_i \f$ for each component \f$ i \f$ of each \f$\vec{k}\f$ vector.
		std::vector<std::deque<std::pair<FP, Vector3D<std::complex<MPCParticleVelocityType> > > > >
			fourierTransformedVelocities;
				/**< The Fourier-transformed velocities measured, along with the
				     corresponding simulation timestamps.*/
}; //class Base
} //namespace FourierTransformedVelocity
} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
