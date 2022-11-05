/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Instrumentation::FourierTransformedVelocity::Simple class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_FOURIERTRANSFORMEDVELOCITY_SIMPLE_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_FOURIERTRANSFORMEDVELOCITY_SIMPLE_HPP

#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/FourierTransformedVelocity/Base.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Simple.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
namespace Instrumentation
{
namespace FourierTransformedVelocity
{
/**
 * Class for measurements of Fourier-transformed velocities in simple MPC fluids.
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
		Simple(const Simulation* const sim, DeviceMemoryManager* devMemMgr, const MPCFluid::Simple* const mpcFluid_);

	private:
		Simple(const Simple&); ///< The copy constructor.

	public:
		/**
		 * The destructor.
		 */
		virtual ~Simple()
		{
		}

	public:
		/**
		 * Performs measurements.
		 */
		virtual void measure();
}; //class Simple
} //namespace FourierTransformedVelocity
} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
