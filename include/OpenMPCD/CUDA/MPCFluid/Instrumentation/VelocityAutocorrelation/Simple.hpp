/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Instrumentation::VelocityAutocorrelation::Simple class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_VELOCITYAUTOCORRELATION_SIMPLE_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_VELOCITYAUTOCORRELATION_SIMPLE_HPP

#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/VelocityAutocorrelation/Base.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
namespace Instrumentation
{
namespace VelocityAutocorrelation
{
/**
 * Class for measurements of velocity autocorrelation in MPC fluids that consist of bare,
 * individual MPC fluid particles.
 */
class Simple : public Base
{
	public:
		/**
		 * The constructor.
		 * @param[in] sim       The simulation instance.
		 * @param[in] devMemMgr The Device memory manager.
		 * @param[in] mpcFluid_ The MPC fluid to measure.
		 */
		Simple(
			const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
			const MPCFluid::Base* const mpcFluid_);

	private:
		Simple(const Simple&); ///< The copy constructor.

	public:
		/**
		 * The destructor.
		 */
		virtual ~Simple()
		{
		}

	protected:
		/**
		 * Populates the currentVelocities buffer with the current velocities of the fluid constituents.
		 */
		virtual void populateCurrentVelocities();
}; //class Simple
} //namespace VelocityAutocorrelation
} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
