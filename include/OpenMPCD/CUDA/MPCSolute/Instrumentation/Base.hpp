/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCSolute::Instrumentation::Base class.
 */

#ifndef OPENMPCD_CUDA_MPCSOLUTE_INSTRUMENTATION_BASE_HPP
#define OPENMPCD_CUDA_MPCSOLUTE_INSTRUMENTATION_BASE_HPP

#include <string>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCSolute
{
/**
 * Namespace for instrumentation classes for MPC solutes.
 */
namespace Instrumentation
{
	/**
	 * Base class for MPC solutes instrumentation.
	 */
	class Base
	{
		protected:
			/**
			 * The constructor.
			 */
			Base()
			{
			}

		private:
			Base(const Base&); ///< The copy constructor.

		public:
			/**
			 * The destructor.
			 */
			virtual ~Base()
			{
			}

		public:
			/**
			 * Performs measurements.
			 */
			void measure()
			{

				measureSpecific();
			}

			/**
			 * Saves the data to the given run directory.
			 * @param[in] rundir The path to the run directory.
			 */
			void save(const std::string& rundir) const
			{

				saveSpecific(rundir);
			}

		protected:
			/**
			 * Performs measurements specific to the solute type.
			 */
			virtual void measureSpecific() = 0;

			/**
			 * Saves measurements specific to the solute type.
			 * @param[in] rundir The path to the run directory.
			 */
			virtual void saveSpecific(const std::string& rundir) const = 0;

		private:
			Base& operator=(const Base&); ///< The assignment operator.

	};
} //namespace Instrumentation
} //namespace MPCSolute
} //namespace CUDA
} //namespace OpenMPCD

#endif
