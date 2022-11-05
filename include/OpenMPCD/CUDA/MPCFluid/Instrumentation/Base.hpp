/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Instrumentation::Base class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_BASE_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_BASE_HPP

#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/FourierTransformedVelocity/Base.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/LogicalEntityMeanSquareDisplacement.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/NormalModeAutocorrelation.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/VelocityAutocorrelation/Base.hpp>

#include <string>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{

class Base;

/**
 * Namespace for instrumentation classes for MPC fluids.
 */
namespace Instrumentation
{
	/**
	 * Base class for MPC fluids instrumentation.
	 */
	class Base
	{
		protected:
			/**
			 * The constructor.
			 *
			 * @param[in] config
			 *            The simulation configuration.
			 *
			 * @param[in] mpcFluid
			 *            The fluid to measure.
			 */
			Base(
				const Configuration& config,
				const CUDA::MPCFluid::Base* const mpcFluid)
				: fourierTransformedVelocity(NULL),
				  logicalEntityMeanSquareDisplacement(NULL),
				  normalModeAutocorrelation(NULL),
				  velocityAutocorrelation(NULL)
			{
				if(LogicalEntityMeanSquareDisplacement::isConfigured(config))
				{
					logicalEntityMeanSquareDisplacement =
						new LogicalEntityMeanSquareDisplacement(
							config, mpcFluid);
				}
				if(NormalModeAutocorrelation::isConfigured(config))
				{
					normalModeAutocorrelation =
						new NormalModeAutocorrelation(config, mpcFluid);
				}
			}

		private:
			Base(const Base&); ///< The copy constructor.

		public:
			/**
			 * The destructor.
			 */
			virtual ~Base()
			{
				delete fourierTransformedVelocity;
				delete logicalEntityMeanSquareDisplacement;
				delete normalModeAutocorrelation;
				delete velocityAutocorrelation;
			}

		public:
			/**
			 * Performs measurements.
			 */
			void measure()
			{
				if(fourierTransformedVelocity)
					fourierTransformedVelocity->measure();

				if(logicalEntityMeanSquareDisplacement)
					logicalEntityMeanSquareDisplacement->measure();

				if(normalModeAutocorrelation)
					normalModeAutocorrelation->measure();

				if(velocityAutocorrelation)
					velocityAutocorrelation->measure();

				measureSpecific();
			}

			/**
			 * Saves the data to the given run directory.
			 * @param[in] rundir The path to the run directory.
			 */
			void save(const std::string& rundir) const
			{
				if(fourierTransformedVelocity)
					fourierTransformedVelocity->save(rundir);

				if(logicalEntityMeanSquareDisplacement)
					logicalEntityMeanSquareDisplacement->save(rundir);

				if(normalModeAutocorrelation)
					normalModeAutocorrelation->save(rundir);

				if(velocityAutocorrelation)
					velocityAutocorrelation->save(rundir);

				saveSpecific(rundir);
			}

			/**
			 * Returns the `LogicalEntityMeanSquareDisplacement` instance, or
			 * `nullptr` if none was configured.
			 */
			const LogicalEntityMeanSquareDisplacement*
			getLogicalEntityMeanSquareDisplacement() const
			{
				return logicalEntityMeanSquareDisplacement;
			}

			/**
			 * Returns the `NormalModeAutocorrelation` instance, or `nullptr` if
			 * none was configured.
			 */
			const NormalModeAutocorrelation*
			getNormalModeAutocorrelation() const
			{
				return normalModeAutocorrelation;
			}

		protected:
			/**
			 * Performs measurements specific to the fluid type.
			 */
			virtual void measureSpecific() = 0;

			/**
			 * Saves measurements specific to the fluid type.
			 * @param[in] rundir The path to the run directory.
			 */
			virtual void saveSpecific(const std::string& rundir) const = 0;

		private:
			Base& operator=(const Base&); ///< The assignment operator.

		protected:
			FourierTransformedVelocity::Base* fourierTransformedVelocity; ///< Measures Fourier-transformed velocities.

			LogicalEntityMeanSquareDisplacement*
				logicalEntityMeanSquareDisplacement;
					/**< Measures the mean square displacement of the centers of
						 mass of the logical entities of the fluid.*/

			NormalModeAutocorrelation* normalModeAutocorrelation;
				///< Measures the autocorrelation of normal mode coordinates.
			VelocityAutocorrelation::Base* velocityAutocorrelation; ///< Measures velocity autocorrelation.
	};
} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
