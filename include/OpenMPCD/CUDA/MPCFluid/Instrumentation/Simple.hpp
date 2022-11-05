/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Instrumentation::Simple class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_SIMPLE_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_SIMPLE_HPP

#include <OpenMPCD/CUDA/MPCFluid/Simple.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/Base.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>

#include <complex>
#include <deque>
#include <vector>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
namespace Instrumentation
{
/**
 * Instrumentation for simple MPC fluids consisting of independent particles.
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
			const MPCFluid::Simple* const mpcFluid_);

		/**
		 * The destructor.
		 */
		virtual ~Simple();

	protected:
		virtual void measureSpecific();

		virtual void saveSpecific(const std::string& rundir) const;
}; //class Simple
} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
