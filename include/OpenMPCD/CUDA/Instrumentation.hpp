/**
 * @file
 * Defines the OpenMPCD::CUDA::Instrumentation class.
 */

#ifndef OPENMPCD_CUDA_INSTRUMENTATION_HPP
#define OPENMPCD_CUDA_INSTRUMENTATION_HPP

#include <OpenMPCD/CUDA/Simulation.hpp>
#include <OpenMPCD/DensityProfile.hpp>
#include <OpenMPCD/FlowProfile.hpp>
#include <OpenMPCD/Graph.hpp>
#include <OpenMPCD/Histogram.hpp>
#include <OpenMPCD/Types.hpp>

#include <boost/scoped_ptr.hpp>

namespace OpenMPCD
{
namespace CUDA
{

/**
 * Class doing measurements in a CUDA simulation.
 */
class Instrumentation
{
	public:
		/**
		 * The constructor.
		 * @param[in] sim          The simulation to measure.
		 * @param[in] rngSeed_     The seed for the random number generator.
		 * @param[in] gitRevision_ A string containing the git revision, if available.
		 */
		Instrumentation(
			const Simulation* const sim, const unsigned int rngSeed_,
			const std::string& gitRevision_);

	private:
		Instrumentation(const Instrumentation&); ///< The copy constructor.

	public:
		/**
		 * The destructor.
		 */
		~Instrumentation();

	public:
		/**
		 * Performs measurements on MPC fluids and solutes.
		 */
		void measure();

		/**
		 * Sets the autosave flag.
		 * @param[in] rundir The directory to save to.
		 */
		void setAutosave(const std::string& rundir)
		{
			autosave=true;
			autosave_rundir = rundir;
		}

		/**
		 * Saves the data to the given run directory.
		 * @param[in] rundir The path to the run directory.
		 */
		void save(const std::string& rundir) const;

	private:
		/**
		 * Performs measurements on the MPC fluid, if there is one.
		 */
		void measureMPCFluid();

		/**
		 * Performs measurements on the MPC solute, if there is one.
		 */
		void measureMPCSolute();

		/**
		 * Saves static data to the given run directory.
		 * @param[in] rundir The path to the run directory.
		 */
		void saveStaticData(const std::string& rundir) const;

	private:
		const Instrumentation& operator=(const Instrumentation&); ///< The assignment operator.

	private:
		const Simulation* const simulation; ///< The simulation to measure
		const unsigned int rngSeed;         ///< The seed for the random number generator.
		const std::string gitRevision;      ///< The string containing the git revision, if available.

		const std::string constructionTimeUTC;
			/**< The UTC time this instance was constructed, as returned by
			 *   `OpenMPCD::Utility::HostInformation::getCurrentUTCTimeAsString`.
			 */

		bool autosave;               ///< Whether to save everything on destruction.
		std::string autosave_rundir; ///< The run directory to save to in case of autosaving.

		unsigned int measurementCount; ///< The number of measurements that have been made.

		boost::scoped_ptr<Histogram> velocityMagnitudeHistogram;
			///< Histogram for the velocity magnitudes.

		boost::scoped_ptr<DensityProfile> densityProfile;
			///< The mass density profile.
		boost::scoped_ptr<FlowProfile<MPCParticleVelocityType> > flowProfile;
			///< The flow profile.

		boost::scoped_ptr<Graph> totalFluidVelocityMagnitudeVSTime;
			/**< Shows the magnitude of the total velocity as a function of
			     time.*/
}; //class Instrumentation
} //namespace CUDA
} //namespace OpenMPCD

#endif
